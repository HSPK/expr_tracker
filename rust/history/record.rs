use super::json::{float_json, number, JsonObject, Step};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule, PyString};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub type SharedLine = Arc<Vec<u8>>;

#[pyfunction]
pub fn history_metric_values(
    py: Python<'_>,
    record: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyDict>>> {
    if !record.is_exact_instance_of::<PyDict>() {
        return Ok(None);
    }
    let metrics = PyDict::new(py);
    for (key, value) in record.cast::<PyDict>()?.iter() {
        if !key.is_instance_of::<PyString>() {
            continue;
        }
        if !key.is_exact_instance_of::<PyString>() {
            return Ok(None);
        }
        let Ok(name) = key.cast::<PyString>()?.to_str() else {
            return Ok(None);
        };
        if !name.starts_with('_') {
            metrics.set_item(key, value)?;
        }
    }
    Ok(Some(metrics.unbind()))
}

pub fn normalize_numpy_scalars(py: Python<'_>, values: &Bound<'_, PyDict>) -> PyResult<()> {
    let mut numpy_type = None;
    let mut checked_numpy = false;
    for (key, value) in values.iter() {
        if !value.is_instance_of::<PyFloat>() || value.is_exact_instance_of::<PyFloat>() {
            continue;
        }
        if !checked_numpy {
            checked_numpy = true;
            let modules = py
                .import("sys")?
                .getattr("modules")?
                .cast_into::<PyDict>()?;
            if let Some(numpy) = modules.get_item("numpy")? {
                if let Ok(numpy) = numpy.cast::<PyModule>() {
                    numpy_type = numpy.dict().get_item("float64")?;
                }
            }
        }
        if numpy_type
            .as_ref()
            .is_some_and(|kind| value.get_type().is(kind))
        {
            values.set_item(key, value.extract::<f64>()?)?;
        }
    }
    Ok(())
}

fn write_float(bytes: &mut Vec<u8>, value: f64) {
    if value.is_nan() {
        bytes.extend_from_slice(b"NaN");
    } else if value == f64::INFINITY {
        bytes.extend_from_slice(b"Infinity");
    } else if value == f64::NEG_INFINITY {
        bytes.extend_from_slice(b"-Infinity");
    } else {
        serde_json::to_writer(bytes, &value).expect("float serialization into a vector");
    }
}

pub fn scalar_object(values: &Bound<'_, PyDict>) -> PyResult<Option<JsonObject>> {
    let mut object = JsonObject::default();
    for (key, value) in values.iter() {
        if !key.is_exact_instance_of::<PyString>() {
            return Ok(None);
        }
        let key = key.cast::<PyString>()?;
        let Ok(name) = key.to_str() else {
            return Ok(None);
        };
        if name.starts_with("_sa") {
            return Ok(None);
        }
        let name = if name.starts_with('\0') {
            format!("\0N{name}")
        } else {
            name.to_owned()
        };
        let text = if value.is_none() {
            "null".into()
        } else if value.is_exact_instance_of::<PyBool>() {
            if value.extract::<bool>()? {
                "true"
            } else {
                "false"
            }
            .into()
        } else if value.is_exact_instance_of::<PyInt>() {
            value.str()?.to_str()?.to_owned()
        } else if value.is_exact_instance_of::<PyFloat>() {
            float_json(value.extract()?)
        } else if value.is_exact_instance_of::<PyString>() {
            let value = value.cast::<PyString>()?;
            let Ok(value) = value.to_str() else {
                return Ok(None);
            };
            serde_json::to_string(value).expect("strings are JSON serializable")
        } else {
            return Ok(None);
        };
        object.set(name, text);
    }
    Ok(Some(object))
}

#[pyclass]
pub struct NativeRecord {
    pub step: Step,
    pub line: SharedLine,
    decoded: Py<PyDict>,
    fresh: bool,
    validated: AtomicBool,
    pub inline_numeric: bool,
    pub timestamp: f64,
}

impl NativeRecord {
    pub fn from_values(
        _py: Python<'_>,
        values: &Bound<'_, PyDict>,
        step: Step,
        timestamp: f64,
        keys: &mut HashMap<String, Vec<u8>>,
    ) -> PyResult<Option<(Self, Vec<(String, String)>)>> {
        for (key, value) in values.iter() {
            let Ok(key) = key.cast::<PyString>() else {
                return Ok(None);
            };
            if !key.is_exact_instance_of::<PyString>()
                || key.to_str().is_err()
                || key.to_str()?.starts_with("_sa")
                || !(value.is_none()
                    || value.is_exact_instance_of::<PyBool>()
                    || value.is_exact_instance_of::<PyInt>()
                    || value.is_exact_instance_of::<PyFloat>()
                    || value.is_exact_instance_of::<PyString>()
                        && value.cast::<PyString>()?.to_str().is_ok())
            {
                return Ok(None);
            }
        }
        let mut bytes = Vec::with_capacity(64 + values.len() * 24);
        bytes.extend_from_slice(b"{\"_step\": ");
        bytes.extend_from_slice(step.0.as_bytes());
        bytes.extend_from_slice(b", \"_time\": ");
        write_float(&mut bytes, timestamp);
        let mut warnings = Vec::new();
        let mut inline_numeric = true;
        for (key, value) in values.iter() {
            let name = key.cast::<PyString>()?.to_str()?;
            if name == "_step" || name == "_time" {
                warnings.push((
                    name.to_owned(),
                    format!("Metric name {name:?} is reserved and will be ignored."),
                ));
                continue;
            }
            bytes.extend_from_slice(b", ");
            if let Some(key) = keys.get(name) {
                bytes.extend_from_slice(key);
            } else if keys.len() < 4096 {
                let key = serde_json::to_vec(name).expect("string serialization");
                bytes.extend_from_slice(&key);
                keys.insert(name.to_owned(), key);
            } else {
                serde_json::to_writer(&mut bytes, name).expect("string serialization");
            }
            bytes.extend_from_slice(b": ");
            if value.is_none() {
                bytes.extend_from_slice(b"null");
            } else if value.is_exact_instance_of::<PyBool>() {
                bytes.extend_from_slice(if value.extract::<bool>()? {
                    b"true"
                } else {
                    b"false"
                });
            } else if value.is_exact_instance_of::<PyInt>() {
                if !name.starts_with('_') && value.extract::<f64>().is_err() {
                    inline_numeric = false;
                }
                bytes.extend_from_slice(value.str()?.to_str()?.as_bytes());
            } else if value.is_exact_instance_of::<PyFloat>() {
                write_float(&mut bytes, value.extract()?);
            } else {
                serde_json::to_writer(&mut bytes, value.cast::<PyString>()?.to_str()?)
                    .expect("string serialization");
            }
        }
        bytes.extend_from_slice(b"}\n");
        Ok(Some((
            Self {
                step,
                timestamp,
                line: Arc::new(bytes),
                decoded: values.clone().unbind(),
                fresh: true,
                validated: AtomicBool::new(true),
                inline_numeric,
            },
            warnings,
        )))
    }

    pub fn new(
        py: Python<'_>,
        object: JsonObject,
        values: &Bound<'_, PyDict>,
        reuse_values: bool,
        fresh: bool,
    ) -> PyResult<Self> {
        let step = object.step().expect("committed records have a step");
        let timestamp = number(object.get("_time").unwrap()).unwrap();
        let mut text = object.encode();
        text.push('\n');
        if fresh {
            return Ok(Self {
                step,
                line: Arc::new(text.into_bytes()),
                decoded: values.clone().unbind(),
                fresh: true,
                validated: AtomicBool::new(false),
                inline_numeric: false,
                timestamp,
            });
        }
        let decoded = PyDict::new(py);
        for (name, raw) in &object.0 {
            let original_name = name.strip_prefix("\0N").unwrap_or(name);
            if reuse_values && name != "_step" && name != "_time" {
                if let Some(value) = values.get_item(original_name)? {
                    decoded.set_item(original_name, value)?;
                    continue;
                }
            }
            if name.starts_with("\0S") {
                return Self::decode_line(py, step, text);
            }
            let value = if raw == "null" {
                py.None()
            } else if raw == "true" || raw == "false" {
                (raw == "true")
                    .into_pyobject(py)?
                    .to_owned()
                    .into_any()
                    .unbind()
            } else if let Ok(integer) = raw.parse::<i64>() {
                integer.into_pyobject(py)?.into_any().unbind()
            } else if matches!(raw.as_str(), "NaN" | "Infinity" | "-Infinity")
                || raw.contains(['.', 'e', 'E']) && !raw.starts_with('"')
            {
                let number = match raw.as_str() {
                    "NaN" => f64::NAN,
                    "Infinity" => f64::INFINITY,
                    "-Infinity" => f64::NEG_INFINITY,
                    _ => match raw.parse::<f64>() {
                        Ok(value) => value,
                        Err(_) => return Self::decode_line(py, step, text),
                    },
                };
                number.into_pyobject(py)?.into_any().unbind()
            } else if raw.starts_with('"') {
                match serde_json::from_str::<String>(raw) {
                    Ok(value) => value.into_pyobject(py)?.into_any().unbind(),
                    Err(_) => return Self::decode_line(py, step, text),
                }
            } else {
                return Self::decode_line(py, step, text);
            };
            decoded.set_item(original_name, value)?;
        }
        Ok(Self {
            step,
            line: Arc::new(text.into_bytes()),
            decoded: decoded.unbind(),
            fresh: false,
            validated: AtomicBool::new(false),
            inline_numeric: false,
            timestamp,
        })
    }

    fn decode_line(py: Python<'_>, step: Step, text: String) -> PyResult<Self> {
        let decoded = py
            .import("json")?
            .getattr("loads")?
            .call1((&text,))?
            .cast_into::<PyDict>()?
            .unbind();
        Ok(Self {
            step,
            line: Arc::new(text.into_bytes()),
            timestamp: decoded.bind(py).get_item("_time")?.unwrap().extract()?,
            decoded,
            fresh: false,
            validated: AtomicBool::new(false),
            inline_numeric: false,
        })
    }
}

impl NativeRecord {
    pub(super) fn validated(&self) -> bool {
        self.validated.load(Ordering::Relaxed)
    }
    pub(super) fn values<'py>(&self, py: Python<'py>) -> &Bound<'py, PyDict> {
        self.decoded.bind(py)
    }
}

#[pymethods]
impl NativeRecord {
    pub(super) fn dictionary<'py>(&self, py: Python<'py>) -> &Bound<'py, PyDict> {
        self.validated.store(false, Ordering::Relaxed);
        self.values(py)
    }
    fn decode(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if !self.fresh {
            return Ok(self.decoded.clone_ref(py));
        }
        let integer = match self.step.0.parse::<i64>() {
            Ok(value) => value.into_pyobject(py)?.into_any(),
            Err(_) => py
                .import("builtins")?
                .getattr("int")?
                .call1((&self.step.0,))?,
        };
        let result = PyDict::new(py);
        result.set_item("_step", &integer)?;
        result.set_item("_time", self.timestamp)?;
        result.update(self.decoded.bind(py).as_mapping())?;
        if self.decoded.bind(py).contains("_step")? {
            result.set_item("_step", &integer)?;
        }
        if self.decoded.bind(py).contains("_time")? {
            result.set_item("_time", self.timestamp)?;
        }
        Ok(result.unbind())
    }
    fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.line)
    }
    fn text(&self) -> &str {
        std::str::from_utf8(&self.line).expect("serialized records are UTF-8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escaped_key_cache_remains_bounded_and_reuses_keys_after_filling() {
        Python::initialize();
        Python::attach(|py| {
            let values = PyDict::new(py);
            let mut keys = HashMap::new();
            let names: Vec<_> = (0..4100)
                .map(|index| format!("metric\"\\\n\0\u{4e2d}_{index}"))
                .collect();
            for name in &names {
                values.set_item(name, 0.5).unwrap();
            }
            values.set_item("_step", 999).unwrap();
            values.set_item("_time", 999).unwrap();
            let (record, warnings) =
                NativeRecord::from_values(py, &values, Step("0".into()), 1.5, &mut keys)
                    .unwrap()
                    .unwrap();
            assert_eq!(keys.len(), 4096);
            assert_eq!(keys[&names[0]], serde_json::to_vec(&names[0]).unwrap());
            assert!(!keys.contains_key(&names[4096]));
            assert_eq!(
                warnings
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
                ["_step", "_time"]
            );
            let decoded: serde_json::Value = serde_json::from_slice(&record.line).unwrap();
            assert_eq!(decoded["_step"], 0);
            assert_eq!(decoded["_time"], 1.5);
            assert_eq!(decoded.as_object().unwrap().len(), 4102);
            for name in &names {
                assert_eq!(decoded[name], 0.5);
            }
            assert!(record.line.ends_with(b"\n"));

            let cached = keys[&names[0]].as_ptr();
            let late = "late\"\\\n\0\u{6587}";
            let next = PyDict::new(py);
            next.set_item(&names[0], 2.0).unwrap();
            next.set_item(late, false).unwrap();
            next.set_item("_hidden", py.None()).unwrap();
            let (record, warnings) =
                NativeRecord::from_values(py, &next, Step("1".into()), 2.5, &mut keys)
                    .unwrap()
                    .unwrap();
            let decoded: serde_json::Value = serde_json::from_slice(&record.line).unwrap();
            assert_eq!(decoded[&names[0]], 2.0);
            assert_eq!(decoded[late], false);
            assert!(decoded["_hidden"].is_null());
            assert!(warnings.is_empty());
            assert_eq!(keys.len(), 4096);
            assert_eq!(keys[&names[0]].as_ptr(), cached);
            assert!(!keys.contains_key(late));
            assert!(!keys.contains_key("_hidden"));
        });
    }

    #[test]
    fn unsupported_scalar_row_does_not_populate_the_key_cache() {
        Python::initialize();
        Python::attach(|py| {
            let values = PyDict::new(py);
            values.set_item("valid", 0.5).unwrap();
            values.set_item("nested", PyDict::new(py)).unwrap();
            let mut keys = HashMap::new();
            assert!(
                NativeRecord::from_values(py, &values, Step("0".into()), 1.0, &mut keys)
                    .unwrap()
                    .is_none()
            );
            assert!(keys.is_empty());
        });
    }
}
