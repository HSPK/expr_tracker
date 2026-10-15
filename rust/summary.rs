use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString};
use std::io::Write;

#[pyclass]
pub struct NativeSummary {
    values: Py<PyDict>,
    pinned: Py<PySet>,
    owner: u32,
}

#[pymethods]
impl NativeSummary {
    #[new]
    fn new(py: Python<'_>) -> Self {
        Self {
            values: PyDict::new(py).unbind(),
            pinned: PySet::empty(py).unwrap().unbind(),
            owner: std::process::id(),
        }
    }

    fn load(&self, py: Python<'_>, values: &Bound<'_, PyDict>) -> PyResult<()> {
        self.check()?;
        self.values.bind(py).update(values.as_mapping())
    }

    fn load_path(&self, py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<()> {
        self.check()?;
        let path = crate::artifacts::path_value(path, false)?;
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        let value = py
            .import("json")?
            .getattr("loads")?
            .call1((pyo3::types::PyBytes::new(py, &bytes),))?;
        if let Ok(values) = value.cast::<PyDict>() {
            self.values.bind(py).update(values.as_mapping())?;
        }
        Ok(())
    }

    pub(crate) fn observe(&self, py: Python<'_>, metrics: &Bound<'_, PyAny>) -> PyResult<()> {
        self.check()?;
        let Ok(metrics) = metrics.cast::<PyDict>() else {
            return Ok(());
        };
        for (key, value) in metrics.iter() {
            let Ok(name) = key.cast::<PyString>() else {
                continue;
            };
            // Python strings may contain unpaired surrogates. Comparing Python
            // objects avoids lossy conversion of those keys.
            let hidden = match name.to_str() {
                Ok(name) => name.starts_with('_'),
                Err(_) => name.call_method1("startswith", ("_",))?.is_truthy()?,
            };
            if hidden {
                continue;
            }
            if self.is_pinned(py, &key)? {
                continue;
            }
            self.values.bind(py).set_item(key, value)?;
        }
        Ok(())
    }

    fn get(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.check()?;
        self.values
            .bind(py)
            .get_item(key)?
            .map(Bound::unbind)
            .ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
    }

    fn set(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check()?;
        self.values.bind(py).set_item(key, value)?;
        self.pinned.bind(py).add(key)?;
        Ok(())
    }

    fn delete(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.check()?;
        self.values.bind(py).del_item(key)?;
        self.pinned.bind(py).discard(key)?;
        Ok(())
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self.check()?;
        Ok(self.values.bind(py).copy()?.unbind())
    }

    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        self.check()?;
        Ok(self.values.bind(py).keys().unbind())
    }

    fn len(&self, py: Python<'_>) -> PyResult<usize> {
        self.check()?;
        Ok(self.values.bind(py).len())
    }

    fn save(
        &self,
        py: Python<'_>,
        path: &Bound<'_, PyAny>,
        payload: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.check()?;
        let path = crate::artifacts::path_value(path, false)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("ensure_ascii", true)?;
        kwargs.set_item("indent", 2)?;
        // Encoding is the SDK boundary: arbitrary Python/numpy values have
        // already passed through RecordCodec before reaching this method.
        let text: String = py
            .import("json")?
            .getattr("dumps")?
            .call((payload,), Some(&kwargs))?
            .extract()?;
        py.detach(|| -> std::io::Result<()> {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| std::path::Path::new("."));
            std::fs::create_dir_all(parent)?;
            let mut file = tempfile::NamedTempFile::new_in(parent)?;
            file.write_all(text.as_bytes())?;
            file.as_file().sync_all()?;
            file.persist(&path).map_err(|e| e.error)?;
            Ok(())
        })?;
        Ok(())
    }
}

impl NativeSummary {
    pub(crate) fn can_inline(&self, py: Python<'_>) -> bool {
        self.pinned
            .bind(py)
            .iter()
            .all(|key| key.is_exact_instance_of::<PyString>())
    }
    fn check(&self) -> PyResult<()> {
        if self.owner != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "This object belongs to another process; create a new run after fork.",
            ));
        }
        Ok(())
    }
    fn is_pinned(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.pinned.bind(py).contains(key)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeSummary>()
}
