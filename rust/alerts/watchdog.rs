use super::{log, seconds, Arc, Condvar, Duration, Instant, Mutex};
use pyo3::prelude::*;

#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertTimer {
    stopping: Arc<(Mutex<TimerState>, Condvar)>,
}
pub(super) struct TimerState {
    stopping: bool,
    alive: bool,
    worker: Option<std::thread::ThreadId>,
}
#[pymethods]
impl AlertTimer {
    #[new]
    pub(super) fn new(callback: Py<PyAny>, interval: f64) -> Self {
        let stopping = Arc::new((
            Mutex::new(TimerState {
                stopping: false,
                alive: interval > 0.0,
                worker: None,
            }),
            Condvar::new(),
        ));
        let shared = stopping.clone();
        if interval > 0.0 {
            std::thread::spawn(move || {
                shared.0.lock().worker = Some(std::thread::current().id());
                loop {
                    let mut state = shared.0.lock();
                    if state.stopping {
                        state.alive = false;
                        shared.1.notify_all();
                        break;
                    }
                    let timeout = shared.1.wait_for(&mut state, seconds(interval)).timed_out();
                    if state.stopping {
                        state.alive = false;
                        shared.1.notify_all();
                        break;
                    }
                    drop(state);
                    if timeout {
                        Python::attach(|py| {
                            if let Err(e) = callback.call0(py) {
                                log(py, format!("Alert watchdog failed: {e}"));
                            }
                        });
                    }
                }
            });
        }
        Self { stopping }
    }
    pub(super) fn close(&self, py: Python<'_>) {
        py.detach(|| {
            let mut state = self.stopping.0.lock();
            state.stopping = true;
            self.stopping.1.notify_all();
            let deadline = Instant::now() + Duration::from_secs(2);
            while state.alive && state.worker != Some(std::thread::current().id()) {
                let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                    break;
                };
                self.stopping.1.wait_for(&mut state, remaining);
            }
        });
    }
    pub(super) fn is_alive(&self) -> bool {
        self.stopping.0.lock().alive
    }
}
