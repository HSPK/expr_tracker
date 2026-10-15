use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[pyclass]
pub struct NativeSpan {
    owner: u32,
    started: Option<Instant>,
    ended: bool,
    started_at: f64,
    duration_ms: f64,
    clock: Option<Py<PyAny>>,
    clock_start: f64,
}

#[pymethods]
impl NativeSpan {
    #[new]
    #[pyo3(signature=(clock=None,started_at=None,clock_start=None))]
    fn new(clock: Option<Py<PyAny>>, started_at: Option<f64>, clock_start: Option<f64>) -> Self {
        Self {
            owner: std::process::id(),
            started: started_at.map(|_| Instant::now()),
            ended: false,
            started_at: started_at.unwrap_or(0.0),
            duration_ms: 0.0,
            clock,
            clock_start: clock_start.unwrap_or(0.0),
        }
    }

    #[pyo3(signature=(clock_start=None))]
    fn begin(&mut self, py: Python<'_>, clock_start: Option<f64>) -> PyResult<(f64, f64)> {
        self.check()?;
        if self.started.is_some() {
            return Err(PyRuntimeError::new_err(
                "A span cannot be begun more than once",
            ));
        }
        self.started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.clock_start = match &self.clock {
            Some(clock) => clock.call0(py)?.extract(py)?,
            None => match clock_start {
                Some(value) => value,
                None => py
                    .import("time")?
                    .getattr("perf_counter")?
                    .call0()?
                    .extract()?,
            },
        };
        self.started = Some(Instant::now());
        Ok((self.started_at, self.clock_start))
    }

    fn end(&mut self, py: Python<'_>) -> PyResult<(bool, f64)> {
        self.check()?;
        if self.ended || self.started.is_none() {
            return Ok((false, self.duration_ms));
        }
        self.ended = true;
        self.duration_ms = match &self.clock {
            Some(clock) => (clock.call0(py)?.extract::<f64>(py)? - self.clock_start) * 1000.0,
            None => self.started.unwrap().elapsed().as_secs_f64() * 1000.0,
        };
        Ok((true, self.duration_ms))
    }

    #[pyo3(signature=(path, depth, track, measured, attributes, error, span_metrics=true, span_count=false))]
    fn record(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        depth: usize,
        track: &Bound<'_, PyAny>,
        measured: &Bound<'_, PyDict>,
        attributes: &Bound<'_, PyDict>,
        error: Option<&Bound<'_, PyAny>>,
        span_metrics: bool,
        span_count: bool,
    ) -> PyResult<Py<PyTuple>> {
        self.check()?;
        let metrics = PyDict::new(py);
        let reductions = PyDict::new(py);
        let record = PyDict::new(py);
        if span_metrics {
            let timing = PyStringBuilder::metric(py, "time_ms", path)?;
            metrics.set_item(timing, self.duration_ms)?;
            if span_count {
                metrics.set_item(PyStringBuilder::metric(py, "count", path)?, 1)?;
            }
            for (key, value) in measured.iter() {
                let metric = py
                    .import("builtins")?
                    .getattr("str")?
                    .call1((&key,))?
                    .call_method1("__add__", ("/",))?
                    .call_method1("__add__", (path,))?;
                metrics.set_item(&metric, value)?;
                if let Ok(name) = key.extract::<String>() {
                    match name.as_str() {
                        "gpu_mem_peak_mb" | "gpu_mem_used_mb" => {
                            reductions.set_item(metric, "max")?
                        }
                        "cpu_percent" | "gpu_percent" => reductions.set_item(metric, "mean")?,
                        _ => {}
                    }
                }
            }
        }
        record.set_item("name", path)?;
        record.set_item("depth", depth)?;
        record.set_item("track", track)?;
        record.set_item("start", self.started_at)?;
        record.set_item("dur_ms", self.duration_ms)?;
        if !measured.is_empty() {
            record.set_item("metrics", measured)?;
        }
        if !attributes.is_empty() {
            record.set_item("args", attributes)?;
        }
        if let Some(error) = error {
            record.set_item("error", error)?;
        }
        Ok(PyTuple::new(
            py,
            [metrics.into_any(), record.into_any(), reductions.into_any()],
        )?
        .unbind())
    }
}

struct PyStringBuilder;
impl PyStringBuilder {
    fn metric<'py>(
        py: Python<'py>,
        metric: &str,
        path: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        use pyo3::types::PyString;
        PyString::new(py, &format!("{metric}/")).call_method1("__add__", (path,))
    }
}
impl NativeSpan {
    pub(crate) fn timing(&self) -> (f64, f64) {
        (self.started_at, self.duration_ms)
    }
    fn check(&self) -> PyResult<()> {
        if self.owner != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "This span belongs to another process; create a new span after fork.",
            ));
        }
        Ok(())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSpan>()
}
