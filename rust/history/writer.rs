use super::json::{JsonObject, Step};
use super::reader::{parse_row, segment_paths, SegmentedReader};
use super::record::SharedLine;
use parking_lot::{Condvar, Mutex};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, IoSlice, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Clone)]
struct Queued {
    step: Step,
    row: i64,
    line: SharedLine,
}
struct Writer {
    path: PathBuf,
    meta_path: PathBuf,
    active: u64,
    active_size: u64,
    buffer_size: usize,
    buffer_interval: Option<f64>,
    max_seconds: Option<f64>,
    max_records: usize,
    max_bytes: Option<usize>,
    max_record: Option<usize>,
    max_file: Option<u64>,
    index_every: usize,
    index: Vec<(Step, usize, u64)>,
    next_anchor: usize,
    run: String,
    async_write: bool,
    queue: VecDeque<Queued>,
    queued_bytes: usize,
    inflight_bytes: usize,
    inflight: usize,
    lines: usize,
    size: u64,
    last: Option<Step>,
    max: Option<Step>,
    flushed: i64,
    sorted: bool,
    duplicates: bool,
    dropped: usize,
    dropped_bytes: usize,
    last_append: Option<Instant>,
    first: Option<Instant>,
    deadline: Option<Instant>,
    requested: bool,
    retry: Option<Instant>,
    closing: bool,
    closed: bool,
    last_meta: Option<Instant>,
    worker_started: bool,
}
impl Writer {
    fn restore(&mut self) -> bool {
        let Some(meta) = fs::read_to_string(&self.meta_path)
            .ok()
            .and_then(|text| JsonObject::parse(&text))
        else {
            return false;
        };
        if meta.get("schema") != Some("2")
            || meta.get("size").and_then(|v| v.parse::<u64>().ok()) != Some(self.size)
        {
            return false;
        }
        let (Some(lines), Some(sorted), Some(duplicates)) = (
            meta.get("lines").and_then(|v| v.parse::<usize>().ok()),
            meta.get("sorted").and_then(|v| match v {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }),
            meta.get("has_duplicate_steps").and_then(|v| match v {
                "true" => Some(true),
                "false" => Some(false),
                _ => None,
            }),
        ) else {
            return false;
        };
        let Some(entries) = meta
            .get("index")
            .and_then(|raw| serde_json::from_str::<Vec<Vec<serde_json::Value>>>(raw).ok())
        else {
            return false;
        };
        let mut anchors = Vec::new();
        let mut previous: Option<(Step, usize, u64)> = None;
        for entry in entries {
            if entry.len() != 3 || !entry[0].is_number() {
                return false;
            }
            let (Some(step), Some(line), Some(offset)) = (
                Step::parse(&entry[0].to_string()),
                entry[1].as_u64(),
                entry[2].as_u64(),
            ) else {
                return false;
            };
            let anchor = (step, line as usize, offset);
            if anchor.1 >= lines
                || anchor.2 >= self.size
                || previous.as_ref().is_some_and(|p| {
                    anchor.1 <= p.1
                        || anchor.2 <= p.2
                        || (sorted && (anchor.0 < p.0 || (!duplicates && anchor.0 == p.0)))
                })
            {
                return false;
            }
            previous = Some(anchor.clone());
            anchors.push(anchor);
        }
        let last = meta
            .get("last_step")
            .filter(|s| *s != "null")
            .and_then(Step::parse);
        let max = meta
            .get("max_step")
            .filter(|s| *s != "null")
            .and_then(Step::parse)
            .or_else(|| last.clone());
        if last
            .as_ref()
            .or(max.as_ref())
            .is_some_and(|s| s.0.len() > 4300)
        {
            return false;
        }
        self.lines = lines;
        self.sorted = sorted;
        self.duplicates = duplicates;
        self.last = last;
        self.max = max;
        self.index = anchors;
        self.next_anchor = self.index.last().map_or(0, |e| e.1 + self.index_every);
        true
    }
    fn capacity(&self, n: usize) -> bool {
        self.queue.len() + self.inflight < self.max_records
            && self
                .max_bytes
                .is_none_or(|max| self.queued_bytes + self.inflight_bytes + n <= max)
    }
    fn schedule(&mut self, immediate: bool) -> bool {
        if self.closing || self.queue.is_empty() {
            return false;
        }
        self.requested |= immediate;
        if self.deadline.is_none() {
            self.deadline = self
                .max_seconds
                .map(|s| self.first.unwrap_or_else(Instant::now) + duration(s));
        }
        if !self.requested && self.deadline.is_none() {
            return false;
        }
        if !self.worker_started {
            self.worker_started = true;
            true
        } else {
            false
        }
    }
    fn track(&mut self, step: Option<Step>, offset: u64) {
        if let Some(step) = step {
            if let Some(last) = &self.last {
                if step < *last {
                    self.sorted = false;
                } else if step == *last {
                    self.duplicates = true;
                }
            }
            if self.lines >= self.next_anchor {
                self.index.push((step.clone(), self.lines, offset));
                self.next_anchor = self.lines + self.index_every;
                if self.index.len() > 4096 {
                    self.index = self.index.iter().step_by(2).cloned().collect();
                    self.index_every *= 2;
                    self.next_anchor = self.index.last().unwrap().1 + self.index_every;
                }
            }
            if self.max.as_ref().is_none_or(|max| step > *max) {
                self.max = Some(step.clone());
            }
            self.last = Some(step);
        }
        self.lines += 1;
    }
    fn active_path(&self) -> PathBuf {
        if self.active == 0 {
            self.path.clone()
        } else {
            PathBuf::from(format!(
                "{}.parts/{:08}.jsonl",
                self.path.display(),
                self.active
            ))
        }
    }
    fn metadata(&self) -> String {
        format!("{{\"schema\":2,\"size\":{},\"lines\":{},\"last_step\":{},\"max_step\":{},\"sorted\":{},\"has_duplicate_steps\":{},\"index\":[{}],\"run\":{}}}",
            self.size,self.lines,self.last.as_ref().map_or("null",|s|&s.0),self.max.as_ref().map_or("null",|s|&s.0),self.sorted,self.duplicates,
            self.index.iter().map(|(s,l,o)|format!("[{},{},{}]",s.0,l,o)).collect::<Vec<_>>().join(","),self.run)
    }
}
fn duration(s: f64) -> Duration {
    Duration::from_secs_f64(if s.is_finite() { s.max(0.0) } else { 0.0 })
}
fn option(value: Option<usize>) -> String {
    value.map_or_else(|| "None".into(), |v| v.to_string())
}
fn callback(call: &Py<PyAny>, _args: ()) -> PyResult<()> {
    Python::attach(|py| call.call0(py).map(|_| ()))
}

fn write_all_slices(writer: &mut impl Write, slices: &mut [IoSlice<'_>]) -> io::Result<()> {
    let mut remaining = slices;
    IoSlice::advance_slices(&mut remaining, 0);
    while !remaining.is_empty() {
        match writer.write_vectored(remaining) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero)),
            Ok(written) => IoSlice::advance_slices(&mut remaining, written),
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[pyclass]
pub struct NativeWriter {
    owner: u32,
    state: Mutex<Writer>,
    condition: Condvar,
    write_lock: Mutex<()>,
    startup: Mutex<Vec<(String, String)>>,
}
impl NativeWriter {
    pub(super) fn snapshot<T>(
        &self,
        f: impl FnOnce(PathBuf, u64, bool, Vec<(Step, usize, u64)>, bool) -> T,
    ) -> T {
        let s = self.state.lock();
        f(
            s.path.clone(),
            s.size,
            s.dropped > 0,
            s.index.clone(),
            s.duplicates || !s.sorted,
        )
    }
    fn check(&self) -> PyResult<()> {
        if self.owner != std::process::id() {
            Err(PyRuntimeError::new_err("Tracker objects cannot be reused after fork; initialize a new run in the child process."))
        } else {
            Ok(())
        }
    }
}
#[pymethods]
impl NativeWriter {
    #[new]
    #[pyo3(signature=(path,buffer_size=50,buffer_interval=Some(1.0),max_buffer_seconds=Some(5.0),max_pending_records=100000,max_pending_bytes=Some(67108864),max_record_bytes=Some(8388608),max_file_bytes=Some(104857600),index_every=1000,meta_extra="{}",async_write=false))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        buffer_size: usize,
        buffer_interval: Option<f64>,
        max_buffer_seconds: Option<f64>,
        max_pending_records: usize,
        max_pending_bytes: Option<usize>,
        max_record_bytes: Option<usize>,
        max_file_bytes: Option<u64>,
        index_every: usize,
        meta_extra: &str,
        async_write: bool,
    ) -> PyResult<Self> {
        let mut state = Writer {
            meta_path: path.with_extension("meta.json"),
            path,
            active: 0,
            active_size: 0,
            buffer_size: buffer_size.max(1),
            buffer_interval,
            max_seconds: max_buffer_seconds,
            max_records: max_pending_records.max(buffer_size).max(1),
            max_bytes: max_pending_bytes,
            max_record: max_record_bytes,
            max_file: max_file_bytes,
            index_every: index_every.max(1),
            index: Vec::new(),
            next_anchor: 0,
            run: meta_extra.into(),
            async_write,
            queue: VecDeque::new(),
            queued_bytes: 0,
            inflight_bytes: 0,
            inflight: 0,
            lines: 0,
            size: 0,
            last: None,
            max: None,
            flushed: -1,
            sorted: true,
            duplicates: false,
            dropped: 0,
            dropped_bytes: 0,
            last_append: None,
            first: None,
            deadline: None,
            requested: false,
            retry: None,
            closing: false,
            closed: false,
            last_meta: None,
            worker_started: false,
        };
        let mut startup = Vec::new();
        py.detach(|| -> io::Result<()> {
            if let Some(parent) = state.path.parent() {
                fs::create_dir_all(parent)?;
            }
            let segments = segment_paths(&state.path)?;
            let active = segments.last().unwrap();
            if *active != state.path {
                state.active = active
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .parse()
                    .unwrap_or(0);
            }
            if active.exists() {
                let mut file = File::open(active)?;
                let size = file.metadata()?.len();
                if size > 0 {
                    file.seek(SeekFrom::Start(size - 1))?;
                    let mut byte = [0];
                    file.read_exact(&mut byte)?;
                    if byte[0] != b'\n' {
                        let mut pos = size;
                        let mut cut = 0;
                        while pos > 0 {
                            let n = pos.min(65536);
                            pos -= n;
                            file.seek(SeekFrom::Start(pos))?;
                            let mut bytes = vec![0; n as usize];
                            file.read_exact(&mut bytes)?;
                            if let Some(index) = bytes.iter().rposition(|b| *b == b'\n') {
                                cut = pos + index as u64 + 1;
                                break;
                            }
                        }
                        match OpenOptions::new()
                            .write(true)
                            .open(active)
                            .and_then(|f| f.set_len(cut))
                        {
                            Ok(()) => startup.push((
                                "warning".into(),
                                format!(
                                    "Dropped {} bytes of a torn final line in {}.",
                                    size - cut,
                                    state.path.display()
                                ),
                            )),
                            Err(error) => startup.push((
                                "error".into(),
                                format!(
                                    "Could not truncate torn tail of {}: {}",
                                    state.path.display(),
                                    error
                                ),
                            )),
                        }
                    }
                }
                state.active_size = file.metadata()?.len();
            }
            let reader = SegmentedReader::open(&state.path)?;
            state.size = reader.size;
            if !state.restore() {
                // Stale unordered prefixes can hide an unindexed maximum.
                // Rebuild fully rather than reuse an unsafe sparse maximum.
                reader.visit(0, true, |row| {
                    state.track(parse_row(&row.bytes).and_then(|r| r.step()), row.offset);
                    Ok(())
                })?;
            }
            state.flushed = state.lines as i64 - 1;
            Ok(())
        })?;
        Ok(Self {
            owner: std::process::id(),
            state: Mutex::new(state),
            condition: Condvar::new(),
            write_lock: Mutex::new(()),
            startup: Mutex::new(startup),
        })
    }
    fn startup_diagnostics(&self) -> Vec<(String, String)> {
        std::mem::take(&mut *self.startup.lock())
    }
    fn get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        self.check()?;
        let s = self.state.lock();
        use pyo3::IntoPyObjectExt;
        match name {
            "path" => s.path.clone().into_py_any(py),
            "meta_path" => s.meta_path.clone().into_py_any(py),
            "buffer_size" => s.buffer_size.into_py_any(py),
            "buffer_interval" => s.buffer_interval.into_py_any(py),
            "max_buffer_seconds" => s.max_seconds.into_py_any(py),
            "max_pending_records" => s.max_records.into_py_any(py),
            "max_pending_bytes" => s.max_bytes.into_py_any(py),
            "max_record_bytes" => s.max_record.into_py_any(py),
            "max_file_bytes" => s.max_file.into_py_any(py),
            "index_every" => s.index_every.into_py_any(py),
            "async_write" => s.async_write.into_py_any(py),
            "lines" => s.lines.into_py_any(py),
            "size" => s.size.into_py_any(py),
            "last_step" => s.last.as_ref().map(|s| s.0.clone()).into_py_any(py),
            "max_step" => s.max.as_ref().map(|s| s.0.clone()).into_py_any(py),
            "flushed_row" => s.flushed.into_py_any(py),
            "sorted" => s.sorted.into_py_any(py),
            "has_duplicate_steps" => s.duplicates.into_py_any(py),
            "dropped" => s.dropped.into_py_any(py),
            "dropped_bytes" => s.dropped_bytes.into_py_any(py),
            "queued_bytes" => s.queued_bytes.into_py_any(py),
            "inflight_bytes" => s.inflight_bytes.into_py_any(py),
            "pending_bytes" => (s.queued_bytes + s.inflight_bytes).into_py_any(py),
            "_closed" => s.closed.into_py_any(py),
            "_closing" => s.closing.into_py_any(py),
            "_inflight_records" => s.inflight.into_py_any(py),
            "_active_number" => s.active.into_py_any(py),
            "_active_size" => s.active_size.into_py_any(py),
            "_active_path" => s.active_path().into_py_any(py),
            "_flush_requested" => s.requested.into_py_any(py),
            "_retry_deadline" => s
                .retry
                .map(|deadline| {
                    deadline
                        .saturating_duration_since(Instant::now())
                        .as_secs_f64()
                })
                .into_py_any(py),
            "index" => s
                .index
                .iter()
                .map(|(s, l, o)| (s.0.clone(), *l, *o))
                .collect::<Vec<_>>()
                .into_py_any(py),
            "buffer" => s
                .queue
                .iter()
                .map(|r| (r.step.0.clone(), r.row, PyBytes::new(py, &r.line).unbind()))
                .collect::<Vec<_>>()
                .into_py_any(py),
            _ => Err(pyo3::exceptions::PyAttributeError::new_err(name.to_owned())),
        }
    }
    fn set(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.check()?;
        let mut s = self.state.lock();
        match name {
            "path" => s.path = value.extract()?,
            "meta_path" => s.meta_path = value.extract()?,
            "buffer_size" => s.buffer_size = value.extract()?,
            "buffer_interval" => s.buffer_interval = value.extract()?,
            "max_buffer_seconds" => s.max_seconds = value.extract()?,
            "max_pending_records" => s.max_records = value.extract()?,
            "max_pending_bytes" => s.max_bytes = value.extract()?,
            "max_record_bytes" => s.max_record = value.extract()?,
            "max_file_bytes" => s.max_file = value.extract()?,
            "index_every" => s.index_every = value.extract()?,
            "async_write" => s.async_write = value.extract()?,
            "has_duplicate_steps" => s.duplicates = value.extract()?,
            "sorted" => s.sorted = value.extract()?,
            "_last_meta_time" => s.last_meta = None,
            _ => return Err(pyo3::exceptions::PyAttributeError::new_err(name.to_owned())),
        }
        Ok(())
    }
    fn enqueue<'a>(
        &'a self,
        py: Python<'_>,
        step: String,
        row: i64,
        line: Vec<u8>,
        start_worker: Py<PyAny>,
        wait_observer: Py<PyAny>,
    ) -> PyResult<(bool, bool, Vec<(String, String)>)> {
        self.enqueue_shared(py, step, row, Arc::new(line), start_worker, wait_observer)
    }
}
impl NativeWriter {
    pub(super) fn flushed(&self) -> i64 {
        self.state.lock().flushed
    }
    pub(super) fn asynchronous(&self) -> bool {
        self.state.lock().async_write
    }
    pub(super) fn enqueue_shared<'a>(
        &'a self,
        py: Python<'_>,
        step: String,
        row: i64,
        line: SharedLine,
        start_worker: Py<PyAny>,
        wait_observer: Py<PyAny>,
    ) -> PyResult<(bool, bool, Vec<(String, String)>)> {
        self.check()?;
        let bytes = line.len();
        let enqueue = |mut s: parking_lot::MutexGuard<'a, Writer>| {
            let mut diagnostics = Vec::new();
            if s.closing {
                diagnostics.push((
                    "warning".into(),
                    format!(
                        "Ignoring write to closing or closed writer {}.",
                        s.path.display()
                    ),
                ));
                return Ok((false, false, diagnostics));
            }
            if s.max_record.is_some_and(|n| bytes > n) || s.max_bytes.is_some_and(|n| bytes > n) {
                s.dropped += 1;
                s.dropped_bytes += bytes;
                diagnostics.push(("error".into(),format!("Rejected oversized record for {}: {} bytes (max_record_bytes={}, max_pending_bytes={}); dropped 1 record.",s.path.display(),bytes,option(s.max_record),option(s.max_bytes))));
                return Ok((false, false, diagnostics));
            }
            while !s.closing && s.retry.is_none() && !s.capacity(bytes) {
                let start = s.schedule(true);
                self.condition.notify_all();
                if start {
                    drop(s);
                    callback(&start_worker, ())?;
                    s = self.state.lock();
                }
                drop(s);
                callback(&wait_observer, ())?;
                s = self.state.lock();
                if !s.closing && s.retry.is_none() && !s.capacity(bytes) {
                    self.condition.wait(&mut s);
                }
            }
            if s.closing {
                diagnostics.push((
                    "warning".into(),
                    format!(
                        "Ignoring write to closing or closed writer {}.",
                        s.path.display()
                    ),
                ));
                return Ok((false, false, diagnostics));
            }
            if !s.capacity(bytes) {
                let mut dropped = 0;
                let mut dropped_bytes = 0;
                while !s.queue.is_empty() && !s.capacity(bytes) {
                    let old = s.queue.pop_front().unwrap();
                    s.queued_bytes -= old.line.len();
                    dropped += 1;
                    dropped_bytes += old.line.len();
                }
                let accepted = s.capacity(bytes);
                if !accepted {
                    dropped += 1;
                    dropped_bytes += bytes;
                }
                s.dropped += dropped;
                s.dropped_bytes += dropped_bytes;
                diagnostics.push(("error".into(),format!("Metrics buffer exceeded {} records or max_pending_bytes={}, dropped {} oldest queued records ({} bytes).",s.max_records,option(s.max_bytes),dropped,dropped_bytes)));
                if !accepted {
                    return Ok((false, false, diagnostics));
                }
            }
            let now = Instant::now();
            let interval = s
                .last_append
                .map(|last| now.duration_since(last).as_secs_f64());
            s.queue.push_back(Queued {
                step: Step(step),
                row,
                line,
            });
            s.queued_bytes += bytes;
            if s.first.is_none() {
                s.first = Some(now);
            }
            s.last_append = Some(now);
            let due = s.queue.len() >= s.buffer_size
                || interval.is_none()
                || s.buffer_interval
                    .is_some_and(|gap| interval.unwrap_or(0.0) >= gap)
                || s.max_seconds
                    .is_some_and(|max| now.duration_since(s.first.unwrap()).as_secs_f64() >= max);
            // Appends do not free capacity or require waking an unchanged timer.
            let before = (s.deadline, s.requested);
            let start = if s.async_write {
                s.schedule(due)
            } else {
                false
            };
            if start || (s.deadline, s.requested) != before {
                self.condition.notify_all();
            }
            drop(s);
            if start {
                callback(&start_worker, ())?;
            }
            Ok((true, due, diagnostics))
        };
        // Keep the guard through cheap admission so capacity cannot race the check.
        let state = self.state.lock();
        if state.capacity(bytes) {
            enqueue(state)
        } else {
            drop(state);
            py.detach(|| enqueue(self.state.lock()))
        }
    }
}

#[path = "writer/persistence.rs"]
mod persistence;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeWriter>()
}

#[cfg(test)]
#[path = "writer/tests.rs"]
mod tests;
