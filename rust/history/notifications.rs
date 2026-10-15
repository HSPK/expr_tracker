use parking_lot::{Condvar, Mutex};
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[pyclass]
pub struct NativeCommitTicket {
    done: Arc<AtomicBool>,
}
struct Queue {
    rows: VecDeque<(Py<PyAny>, Arc<AtomicBool>)>,
    owner: Option<u64>,
    depth: usize,
}
#[pyclass]
pub struct NativeCommitQueue {
    state: Mutex<Queue>,
    condition: Condvar,
}
#[pymethods]
impl NativeCommitQueue {
    #[new]
    fn new() -> Self {
        Self {
            state: Mutex::new(Queue {
                rows: VecDeque::new(),
                owner: None,
                depth: 0,
            }),
            condition: Condvar::new(),
        }
    }
    fn enqueue(
        &self,
        py: Python<'_>,
        mut rows: Vec<Py<PyAny>>,
    ) -> PyResult<Option<Py<NativeCommitTicket>>> {
        let Some(last) = rows.pop() else {
            return Ok(None);
        };
        let done = Arc::new(AtomicBool::new(false));
        let ticket = Py::new(py, NativeCommitTicket { done: done.clone() })?;
        let mut state = self.state.lock();
        for row in rows {
            state
                .rows
                .push_back((row, Arc::new(AtomicBool::new(false))));
        }
        state.rows.push_back((last, done));
        self.condition.notify_all();
        Ok(Some(ticket))
    }
    #[pyo3(signature=(owner,notify,target=None))]
    fn drain(
        &self,
        py: Python<'_>,
        owner: u64,
        notify: Py<PyAny>,
        target: Option<PyRef<'_, NativeCommitTicket>>,
    ) -> PyResult<()> {
        let target = target.map(|ticket| ticket.done.clone()).or_else(|| {
            let state = self.state.lock();
            // Recursive drains stop at the queued tail captured on entry.
            if state.owner == Some(owner) {
                state.rows.back().map(|(_, done)| done.clone())
            } else {
                None
            }
        });
        while let Some((object, done)) = self.take_next(py, owner, &target) {
            let result = notify.call1(py, (object,));
            self.finish(&done);
            result?;
        }
        Ok(())
    }
    #[getter]
    fn count(&self) -> usize {
        self.state.lock().rows.len()
    }
    #[getter]
    fn owner(&self) -> Option<u64> {
        self.state.lock().owner
    }
    #[getter]
    fn depth(&self) -> usize {
        self.state.lock().depth
    }
}
impl NativeCommitQueue {
    pub(super) fn idle(&self) -> bool {
        let state = self.state.lock();
        state.owner.is_none() && state.rows.is_empty()
    }
    fn take_next(
        &self,
        py: Python<'_>,
        owner: u64,
        target: &Option<Arc<AtomicBool>>,
    ) -> Option<(Py<PyAny>, Arc<AtomicBool>)> {
        let poll = |mut state: parking_lot::MutexGuard<'_, Queue>| loop {
            if target
                .as_ref()
                .is_some_and(|done| done.load(Ordering::Acquire))
            {
                return None;
            }
            if target.is_none()
                && state.rows.is_empty()
                && (state.owner.is_none() || state.owner == Some(owner))
            {
                return None;
            }
            if !state.rows.is_empty() && (state.owner.is_none() || state.owner == Some(owner)) {
                state.owner = Some(owner);
                state.depth += 1;
                return state.rows.pop_front();
            }
            self.condition.wait(&mut state);
        };
        let state = self.state.lock();
        // Release the GIL only when another notification must finish first.
        let ready = target
            .as_ref()
            .is_some_and(|done| done.load(Ordering::Acquire))
            || ((state.owner.is_none() || state.owner == Some(owner))
                && (!state.rows.is_empty() || target.is_none()));
        if ready {
            poll(state)
        } else {
            drop(state);
            py.detach(|| poll(self.state.lock()))
        }
    }
    fn finish(&self, done: &Arc<AtomicBool>) {
        let mut state = self.state.lock();
        done.store(true, Ordering::Release);
        state.depth -= 1;
        if state.depth == 0 {
            state.owner = None;
        }
        self.condition.notify_all();
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCommitTicket>()?;
    module.add_class::<NativeCommitQueue>()
}
