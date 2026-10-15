use super::{
    attr, text_in, text_out, Arc, AtomicBool, Condvar, Duration, HashMap, Instant, Mutex, Ordering,
    PyDict, PyRuntimeError, VecDeque,
};
use pyo3::prelude::*;

#[pyfunction]
pub(super) fn alert_accepts(
    config: &Bound<'_, PyAny>,
    message: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !attr::<bool>(config, "enabled")? {
        return Ok(false);
    }
    let level = message.getattr("level")?;
    let levels = config.getattr("levels")?;
    if !levels.is_none() {
        if !levels.contains(&level)? {
            return Ok(false);
        }
    } else if attr::<usize>(&level, "rank")? < attr::<usize>(&config.getattr("min_level")?, "rank")?
    {
        return Ok(false);
    }
    let tags = config.getattr("tags")?;
    if tags.is_truthy()? {
        let message_tags = message.getattr("tags")?;
        let mut matches = false;
        for tag in tags.try_iter()? {
            if message_tags.contains(&tag?)? {
                matches = true;
                break;
            }
        }
        if !matches {
            return Ok(false);
        }
    }
    Ok(true)
}
#[pyfunction]
#[pyo3(signature=(names,requested=None))]
pub(super) fn alert_select_channels(
    py: Python<'_>,
    names: Vec<Py<PyAny>>,
    requested: Option<Vec<Py<PyAny>>>,
) -> PyResult<(Vec<Py<PyAny>>, Vec<Py<PyAny>>)> {
    if let Some(requested) = requested.filter(|r| !r.is_empty()) {
        let mut selected = vec![];
        let mut missing = vec![];
        for name in requested {
            let mut found = false;
            for known in &names {
                if known.bind(py).eq(name.bind(py))? {
                    found = true;
                    break;
                }
            }
            if found {
                selected.push(name);
            } else {
                missing.push(name);
            }
        }
        Ok((selected, missing))
    } else {
        Ok((names, vec![]))
    }
}
pub(super) fn seconds(d: f64) -> Duration {
    Duration::from_secs_f64(if d.is_finite() { d.max(0.0) } else { 0.0 })
}
pub(super) fn log(py: Python<'_>, text: String) {
    if let Ok(logger) = py.import("loguru").and_then(|m| m.getattr("logger")) {
        if let Ok(text) = text_out(py, &text) {
            let _ = logger.call_method1("warning", (text,));
        }
    }
}

pub(super) struct Bucket {
    capacity: f64,
    tokens: f64,
    updated: Instant,
}
impl Bucket {
    pub(super) fn new(rate: Option<f64>) -> Self {
        let capacity = rate.unwrap_or(0.0);
        Self {
            capacity,
            tokens: capacity,
            updated: Instant::now(),
        }
    }
    pub(super) fn acquire(&mut self) -> bool {
        if self.capacity <= 0.0 {
            return true;
        }
        let now = Instant::now();
        self.tokens = (self.tokens
            + now.duration_since(self.updated).as_secs_f64() * self.capacity / 60.0)
            .min(self.capacity);
        self.updated = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}
#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertTokenBucket {
    inner: Mutex<Bucket>,
}
#[pymethods]
impl AlertTokenBucket {
    #[new]
    pub(super) fn new(rate_per_minute: Option<f64>) -> Self {
        Self {
            inner: Mutex::new(Bucket::new(rate_per_minute)),
        }
    }
    pub(super) fn acquire(&self) -> bool {
        self.inner.lock().acquire()
    }
}
pub(super) struct Dedup {
    window: f64,
    seen: HashMap<String, (f64, usize)>,
    origin: Instant,
}
impl Dedup {
    pub(super) fn new(window: f64) -> Self {
        Self {
            window,
            seen: HashMap::new(),
            origin: Instant::now(),
        }
    }
    pub(super) fn check(&mut self, key: String, now: Option<f64>) -> (bool, usize) {
        if self.window <= 0.0 {
            return (true, 0);
        }
        let now = now.unwrap_or_else(|| self.origin.elapsed().as_secs_f64());
        let old = self.seen.get(&key).copied();
        match old {
            Some((last, count)) if !(now - last >= self.window) => {
                self.seen.insert(key, (last, count + 1));
                (false, count + 1)
            }
            _ => {
                self.seen.insert(key, (now, 0));
                if self.seen.len() > 10000 {
                    self.seen
                        .retain(|_, (last, _)| !(now - *last >= self.window));
                }
                (true, old.map_or(0, |(_, count)| count))
            }
        }
    }
}
#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertDeduper {
    inner: Mutex<Dedup>,
}
#[pymethods]
impl AlertDeduper {
    #[new]
    pub(super) fn new(window: f64) -> Self {
        Self {
            inner: Mutex::new(Dedup::new(window)),
        }
    }
    #[pyo3(signature=(key,now=None))]
    pub(super) fn check(
        &self,
        key: &Bound<'_, PyAny>,
        now: Option<f64>,
    ) -> PyResult<(bool, usize)> {
        Ok(self.inner.lock().check(text_in(key)?, now))
    }
    pub(super) fn __len__(&self) -> usize {
        self.inner.lock().seen.len()
    }
}
#[derive(Clone)]
pub(super) struct Policy {
    asynchronous: bool,
    queue_size: usize,
    full: String,
    rate_action: String,
    rate: Option<f64>,
    retries: usize,
    initial: f64,
    factor: f64,
    max: f64,
    respect: bool,
    silent: bool,
}
impl Policy {
    pub(super) fn from_py(p: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            asynchronous: attr(p, "async_send")?,
            queue_size: attr::<i64>(p, "queue_size")?.max(1) as usize,
            full: attr(p, "on_queue_full")?,
            rate_action: attr(p, "on_rate_limited")?,
            rate: attr(p, "rate_limit_per_minute")?,
            retries: attr::<i64>(p, "max_retries")?.max(0) as usize,
            initial: attr(p, "backoff_initial")?,
            factor: attr(p, "backoff_factor")?,
            max: attr(p, "backoff_max")?,
            respect: attr(p, "respect_retry_after")?,
            silent: attr(p, "fail_silently")?,
        })
    }
}
pub(super) struct LaneState {
    queue: VecDeque<Py<PyAny>>,
    inflight: usize,
    sent: usize,
    failed: usize,
    suppressed: usize,
    dropped: usize,
    alive: bool,
    worker: Option<std::thread::ThreadId>,
}
pub(super) struct Lane {
    state: Mutex<LaneState>,
    wake: Condvar,
    done: Condvar,
    stopping: AtomicBool,
    bucket: Mutex<Bucket>,
    dedup: Mutex<Dedup>,
    policy: Policy,
    backend: Py<PyAny>,
    config: Py<PyAny>,
}
impl Lane {
    pub(super) fn delivery(&self, py: Python<'_>, message: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = text_in(&message.call_method0("key")?)?;
        let (allowed, suppressed) = self.dedup.lock().check(key, None);
        if !allowed {
            self.state.lock().suppressed += 1;
            return Ok(());
        }
        if !self.bucket.lock().acquire() {
            if self.policy.rate_action != "queue" {
                self.state.lock().suppressed += 1;
                return Ok(());
            }
            let interval = 60.0 / self.policy.rate.unwrap_or(1.0).max(1.0);
            let deadline = Instant::now() + seconds(interval.max(1.0) * 60.0);
            while !self.bucket.lock().acquire() {
                if Instant::now() >= deadline || self.stopping.load(Ordering::Acquire) {
                    self.state.lock().suppressed += 1;
                    return Ok(());
                }
                py.detach(|| std::thread::sleep(seconds(interval.min(0.5))));
            }
        }
        let payload = if suppressed > 0 {
            let kwargs = message.call_method0("to_dict")?;
            let kwargs = kwargs.cast::<PyDict>()?;
            kwargs.set_item("level", message.getattr("level")?)?;
            let clone = message.get_type().call((), Some(kwargs))?;
            let text = text_in(&message.getattr("text")?)?;
            clone.setattr(
                "text",
                text_out(
                    py,
                    &format!("{text}\n(+{suppressed} similar alerts suppressed)"),
                )?,
            )?;
            clone
        } else {
            message.clone()
        };
        let mut delay = self.policy.initial;
        let mut last = None;
        for attempt in 0..=self.policy.retries {
            match self.backend.bind(py).call_method1("send", (&payload,)) {
                Ok(_) => {
                    self.state.lock().sent += 1;
                    return Ok(());
                }
                Err(err) => {
                    let value = err.value(py);
                    let retryable = attr::<bool>(value, "retryable").unwrap_or(true);
                    let wait = if self.policy.respect {
                        attr::<Option<f64>>(value, "retry_after")
                            .unwrap_or(None)
                            .filter(|v| *v != 0.0)
                    } else {
                        None
                    };
                    last = Some(err);
                    if attempt == self.policy.retries || !retryable {
                        break;
                    }
                    let jitter = py
                        .import("random")?
                        .call_method0("random")?
                        .extract::<f64>()?;
                    let sleep = wait
                        .unwrap_or(delay * (1.0 + jitter * 0.25))
                        .min(self.policy.max);
                    py.detach(|| std::thread::sleep(seconds(sleep)));
                    delay = (delay * self.policy.factor).min(self.policy.max);
                }
            }
        }
        self.state.lock().failed += 1;
        let name = self.config.bind(py).getattr("name")?.repr()?.to_string();
        let last = last.unwrap();
        let text = format!("Failed to send alert via {name:?}: {last}");
        if self.policy.silent {
            log(py, text);
            Ok(())
        } else {
            let err = PyRuntimeError::new_err(text);
            err.set_cause(py, Some(last));
            Err(err)
        }
    }
    pub(super) fn start(self: &Arc<Self>) {
        let mut state = self.state.lock();
        if state.alive {
            return;
        }
        self.stopping.store(false, Ordering::Release);
        state.alive = true;
        let shared = self.clone();
        std::thread::spawn(move || {
            shared.state.lock().worker = Some(std::thread::current().id());
            loop {
                let message = {
                    let mut state = shared.state.lock();
                    while state.queue.is_empty() && !shared.stopping.load(Ordering::Acquire) {
                        shared.wake.wait(&mut state);
                    }
                    if state.queue.is_empty() && shared.stopping.load(Ordering::Acquire) {
                        state.alive = false;
                        state.worker = None;
                        shared.done.notify_all();
                        break;
                    }
                    let message = state.queue.pop_front().unwrap();
                    state.inflight += 1;
                    shared.done.notify_all();
                    message
                };
                Python::attach(|py| {
                    if let Err(e) = shared.delivery(py, message.bind(py)) {
                        log(py, format!("Alert dispatch worker error: {e}"));
                    }
                });
                let mut state = shared.state.lock();
                state.inflight -= 1;
                shared.done.notify_all();
            }
        });
    }
    pub(super) fn flush(&self, timeout: f64) {
        let deadline = Instant::now() + seconds(timeout);
        let mut state = self.state.lock();
        while (!state.queue.is_empty() || state.inflight > 0)
            && state.worker != Some(std::thread::current().id())
        {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            self.done.wait_for(&mut state, remaining);
        }
    }
}
#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertChannel {
    inner: Arc<Lane>,
}
#[pymethods]
impl AlertChannel {
    #[new]
    pub(super) fn new(
        py: Python<'_>,
        config: Py<PyAny>,
        backend: Py<PyAny>,
        policy: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let policy_native = Policy::from_py(policy)?;
        Ok(Self {
            inner: Arc::new(Lane {
                state: Mutex::new(LaneState {
                    queue: VecDeque::new(),
                    inflight: 0,
                    sent: 0,
                    failed: 0,
                    suppressed: 0,
                    dropped: 0,
                    alive: false,
                    worker: None,
                }),
                wake: Condvar::new(),
                done: Condvar::new(),
                stopping: AtomicBool::new(false),
                bucket: Mutex::new(Bucket::new(policy_native.rate)),
                dedup: Mutex::new(Dedup::new(attr(policy, "dedup_window")?)),
                policy: policy_native,
                backend: backend.clone_ref(py),
                config,
            }),
        })
    }
    pub(super) fn accepts(&self, py: Python<'_>, message: &Bound<'_, PyAny>) -> PyResult<bool> {
        alert_accepts(self.inner.config.bind(py), message)
    }
    pub(super) fn send(&self, py: Python<'_>, message: Py<PyAny>) -> PyResult<()> {
        if !self.inner.policy.asynchronous {
            return self.inner.delivery(py, message.bind(py));
        }
        self.inner.start();
        let mut state = self.inner.state.lock();
        if state.queue.len() >= self.inner.policy.queue_size {
            match self.inner.policy.full.as_str() {
                "block" => {
                    drop(state);
                    py.detach(|| {
                        let mut state = self.inner.state.lock();
                        while state.queue.len() >= self.inner.policy.queue_size {
                            self.inner.done.wait(&mut state);
                        }
                        state.queue.push_back(message);
                        self.inner.wake.notify_one();
                    });
                    return Ok(());
                }
                "drop_oldest" => {
                    state.queue.pop_front();
                    state.dropped += 1;
                }
                _ => {
                    state.dropped += 1;
                    drop(state);
                    log(py, "Alert queue is full; dropping.".into());
                    return Ok(());
                }
            }
        }
        state.queue.push_back(message);
        self.inner.wake.notify_one();
        Ok(())
    }
    pub(super) fn deliver(&self, py: Python<'_>, message: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.delivery(py, message)
    }
    pub(super) fn start(&self) {
        self.inner.start();
    }
    pub(super) fn is_worker_thread(&self) -> bool {
        self.inner.state.lock().worker == Some(std::thread::current().id())
    }
    #[pyo3(signature=(timeout=5.0))]
    pub(super) fn flush(&self, py: Python<'_>, timeout: f64) {
        py.detach(|| self.inner.flush(timeout));
    }
    #[pyo3(signature=(timeout=5.0))]
    pub(super) fn close(&self, py: Python<'_>, timeout: f64) {
        let worker = self.is_worker_thread();
        py.detach(|| {
            self.inner.flush(timeout);
            self.inner.stopping.store(true, Ordering::Release);
            self.inner.wake.notify_all();
            if !worker {
                let deadline = Instant::now() + seconds(timeout);
                let mut state = self.inner.state.lock();
                while state.alive {
                    let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                        break;
                    };
                    self.inner.done.wait_for(&mut state, remaining);
                }
            }
        });
    }
    pub(super) fn stats(&self) -> (usize, usize, usize, usize, usize, bool, usize) {
        let s = self.inner.state.lock();
        (
            s.sent,
            s.failed,
            s.suppressed,
            s.dropped,
            s.queue.len(),
            s.alive,
            s.queue.len() + s.inflight,
        )
    }
}

#[pyfunction]
#[pyo3(signature=(channels,timeout=5.0))]
pub(super) fn alert_flush_channels(py: Python<'_>, channels: Vec<Py<AlertChannel>>, timeout: f64) {
    let lanes: Vec<_> = channels
        .iter()
        .map(|channel| channel.borrow(py).inner.clone())
        .collect();
    py.detach(|| {
        let deadline = Instant::now() + seconds(timeout);
        for lane in lanes {
            lane.flush(
                deadline
                    .saturating_duration_since(Instant::now())
                    .as_secs_f64(),
            );
        }
    });
}
