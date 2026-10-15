use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyString};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

type Point = (String, f64, f64);
struct SampleTime {
    step: String,
    time: f64,
}
type StoredPoint = (Arc<SampleTime>, f64);
fn export(point: &StoredPoint) -> Point {
    (point.0.step.clone(), point.0.time, point.1)
}
fn push_point(points: &mut VecDeque<StoredPoint>, window: usize, point: StoredPoint) {
    if points.len() >= window.saturating_mul(2) {
        points.drain(..points.len() + 1 - window);
    }
    points.push_back(point);
}
fn ascii_key(name: &str) -> String {
    let raw = serde_json::to_string(name).unwrap();
    let mut encoded = String::with_capacity(raw.len());
    for character in raw.chars() {
        if character.is_ascii() && character != '\u{7f}' {
            encoded.push(character);
        } else {
            use std::fmt::Write;
            for unit in character.encode_utf16(&mut [0; 2]) {
                write!(&mut encoded, "\\u{unit:04x}").unwrap();
            }
        }
    }
    encoded
}
struct Series {
    window: usize,
    values: HashMap<String, VecDeque<StoredPoint>>,
    non_numeric: HashSet<String>,
    key_cache: HashMap<String, String>,
}
#[pyclass]
pub struct NativeMetricSeries {
    state: Mutex<Series>,
}
impl NativeMetricSeries {
    /// Keys use the SDK's ASCII JSON string encoding, including surrounding
    /// quotes, so unpaired-surrogate metric names remain lossless.
    pub fn has_json_key(&self, key: &str) -> bool {
        self.has(key)
    }
    pub fn latest_json_key(&self, key: &str) -> Option<(String, f64, f64)> {
        self.latest(key)
    }
    pub fn window_json_key(
        &self,
        key: &str,
        count: Option<i64>,
        duration: Option<f64>,
        now: Option<f64>,
    ) -> Option<Vec<(String, f64, f64)>> {
        self.window_points(key, count, duration, now)
    }
}
#[pymethods]
impl NativeMetricSeries {
    #[new]
    #[pyo3(signature=(window=4096))]
    fn new(window: usize) -> Self {
        Self {
            state: Mutex::new(Series {
                window: window.max(2),
                values: HashMap::new(),
                non_numeric: HashSet::new(),
                key_cache: HashMap::new(),
            }),
        }
    }
    #[getter]
    fn window(&self) -> usize {
        self.state.lock().window
    }
    fn ensure_capacity(&self, window: usize) {
        let mut s = self.state.lock();
        s.window = s.window.max(window);
    }
    fn add(
        &self,
        step: String,
        time: f64,
        values: Vec<(String, Option<f64>, bool)>,
    ) -> Vec<String> {
        let mut s = self.state.lock();
        let mut warnings = Vec::new();
        let window = s.window;
        let sample = Arc::new(SampleTime { step, time });
        for (key, value, failed) in values {
            if key.starts_with('_') {
                continue;
            }
            if let Some(value) = value {
                let series = s.values.entry(key).or_default();
                push_point(series, window, (sample.clone(), value));
            } else {
                if failed && !s.non_numeric.contains(&key) {
                    warnings.push(key.clone());
                }
                s.non_numeric.insert(key);
            }
        }
        warnings
    }
    pub(crate) fn add_record(
        &self,
        py: Python<'_>,
        step: String,
        time: f64,
        record: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<String>> {
        for (key, value) in record.iter() {
            if !key.is_exact_instance_of::<PyString>()
                || key.cast::<PyString>()?.to_str().is_err()
                || (value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>())
                    && !value.is_exact_instance_of::<PyFloat>()
                    && !value.is_exact_instance_of::<PyInt>()
                    && !value.is_exact_instance_of::<PyBool>()
            {
                return self.add_record_compatible(py, step, time, record);
            }
        }
        self.add_validated_record(py, step, time, record)
    }
    fn names(&self) -> HashSet<String> {
        let s = self.state.lock();
        s.values
            .keys()
            .chain(s.non_numeric.iter())
            .cloned()
            .collect()
    }
    fn has(&self, name: &str) -> bool {
        let s = self.state.lock();
        s.values.contains_key(name) || s.non_numeric.contains(name)
    }
    fn points(&self, name: &str) -> Vec<Point> {
        let s = self.state.lock();
        s.values.get(name).map_or_else(Vec::new, |v| {
            v.iter()
                .skip(v.len().saturating_sub(s.window))
                .map(export)
                .collect()
        })
    }
    #[pyo3(signature=(name,count=None,duration=None,now=None))]
    fn window_points(
        &self,
        name: &str,
        count: Option<i64>,
        duration: Option<f64>,
        now: Option<f64>,
    ) -> Option<Vec<Point>> {
        let s = self.state.lock();
        let Some(series) = s.values.get(name) else {
            return None;
        };
        let count = count.unwrap_or(s.window as i64);
        let mut start = (series.len() as i128 - count as i128)
            .max(0)
            .min(series.len() as i128) as usize;
        if let Some(duration) = duration {
            let cutoff = now.unwrap_or(series.back().unwrap().0.time) - duration;
            let mut index = series.len();
            while index > start && series[index - 1].0.time >= cutoff {
                index -= 1;
            }
            start = index;
        }
        Some(series.iter().skip(start).map(export).collect())
    }
    fn latest(&self, name: &str) -> Option<Point> {
        self.state
            .lock()
            .values
            .get(name)
            .and_then(|v| v.back())
            .map(export)
    }
    fn clear(&self) {
        let mut s = self.state.lock();
        s.values.clear();
        s.non_numeric.clear();
    }
}
impl NativeMetricSeries {
    pub(crate) fn add_validated_record(
        &self,
        py: Python<'_>,
        step: String,
        time: f64,
        record: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<String>> {
        let mut s = self.state.lock();
        let Series {
            window,
            values,
            non_numeric,
            key_cache,
        } = &mut *s;
        let sample = Arc::new(SampleTime { step, time });
        let mut warnings = Vec::new();
        for (key, value) in record.iter() {
            let name = key.cast::<PyString>()?.to_str()?;
            if name.starts_with('_') {
                continue;
            }
            let temporary;
            let key: &String = if let Some(key) = key_cache.get(name) {
                key
            } else if key_cache.len() < 4096 {
                key_cache
                    .entry(name.to_owned())
                    .or_insert_with(|| ascii_key(name))
            } else {
                temporary = ascii_key(name);
                &temporary
            };
            let numeric = value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>();
            let (number, failed) = if numeric {
                match value.extract::<f64>() {
                    Ok(value) => (Some(value), false),
                    Err(error) if error.is_instance_of::<pyo3::exceptions::PyOverflowError>(py) => {
                        (None, true)
                    }
                    Err(error) => return Err(error),
                }
            } else {
                (None, false)
            };
            if let Some(number) = number {
                let points = if let Some(points) = values.get_mut(key) {
                    points
                } else {
                    values.entry(key.clone()).or_default()
                };
                push_point(points, *window, (sample.clone(), number));
            } else if !non_numeric.contains(key) {
                if failed {
                    warnings.push(key.clone());
                }
                non_numeric.insert(key.clone());
            }
        }
        Ok(warnings)
    }
    fn add_record_compatible(
        &self,
        py: Python<'_>,
        step: String,
        time: f64,
        record: &Bound<'_, PyDict>,
    ) -> PyResult<Vec<String>> {
        let dumps = py.import("json")?.getattr("dumps")?;
        let float = py.import("builtins")?.getattr("float")?;
        let mut normalized = Vec::new();
        for (key, value) in record.iter() {
            if key.call_method1("startswith", ("_",))?.is_truthy()? {
                continue;
            }
            let name = dumps.call1((&key,))?.extract()?;
            let numeric = value.is_instance_of::<PyFloat>() || value.is_instance_of::<PyInt>();
            let (number, failed) = if numeric {
                match float
                    .call1((&value,))
                    .and_then(|value| value.extract::<f64>())
                {
                    Ok(value) => (Some(value), false),
                    Err(error)
                        if error.is_instance_of::<pyo3::exceptions::PyOverflowError>(py)
                            || error.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                            || error.is_instance_of::<pyo3::exceptions::PyValueError>(py) =>
                    {
                        (None, true)
                    }
                    Err(error) => return Err(error),
                }
            } else {
                (None, false)
            };
            normalized.push((name, number, failed));
        }
        Ok(self.add(step, time, normalized))
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeMetricSeries>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validated_key_cache_is_bounded_and_reuses_ascii_encoding_after_saturation() {
        Python::initialize();
        Python::attach(|py| {
            let series = NativeMetricSeries::new(2);
            let record = PyDict::new(py);
            let names: Vec<_> = (0..4100)
                .map(|index| format!("metric\"\\\n\0\u{7f}\u{4e2d}\u{1f680}_{index}"))
                .collect();
            let dumps = py.import("json").unwrap().getattr("dumps").unwrap();
            let encoded: Vec<String> = names
                .iter()
                .map(|name| dumps.call1((name,)).unwrap().extract().unwrap())
                .collect();
            for (index, name) in names.iter().enumerate() {
                record.set_item(name, index as f64).unwrap();
            }
            record.set_item("_hidden", py.None()).unwrap();
            assert!(series
                .add_validated_record(py, "0".into(), 1.5, &record)
                .unwrap()
                .is_empty());
            let cached = {
                let state = series.state.lock();
                assert_eq!(state.key_cache.len(), 4096);
                assert!(!state.key_cache.contains_key("_hidden"));
                for (index, (name, key)) in names.iter().zip(&encoded).enumerate() {
                    if index < 4096 {
                        assert_eq!(&state.key_cache[name], key);
                    } else {
                        assert!(!state.key_cache.contains_key(name));
                    }
                    assert_eq!(
                        export(&state.values[key][0]),
                        ("0".into(), 1.5, index as f64)
                    );
                }
                state.key_cache[&names[0]].as_ptr()
            };
            let next = PyDict::new(py);
            next.set_item(&names[0], 2.0).unwrap();
            next.set_item("late\u{6587}", 3.0).unwrap();
            next.set_item("text", "value").unwrap();
            next.set_item("huge", py.eval(c"10 ** 1000", None, None).unwrap())
                .unwrap();
            assert_eq!(
                series
                    .add_validated_record(py, "1".into(), 2.5, &next)
                    .unwrap(),
                [ascii_key("huge")]
            );
            assert!(series
                .add_validated_record(py, "2".into(), 3.5, &next)
                .unwrap()
                .is_empty());
            {
                let state = series.state.lock();
                assert_eq!(state.key_cache.len(), 4096);
                assert_eq!(state.key_cache[&names[0]].as_ptr(), cached);
                assert!(!state.key_cache.contains_key("late\u{6587}"));
                assert!(state.non_numeric.contains(&ascii_key("text")));
                assert!(state.non_numeric.contains(&ascii_key("huge")));
                assert_eq!(
                    export(state.values[&encoded[0]].back().unwrap()),
                    ("2".into(), 3.5, 2.0)
                );
            }
            series.clear();
            assert_eq!(series.state.lock().key_cache.len(), 4096);
            assert_eq!(
                series
                    .add_validated_record(py, "3".into(), 4.5, &next)
                    .unwrap(),
                [ascii_key("huge")]
            );
        });
    }

    #[test]
    fn conversion_error_preserves_current_key_cache_and_prior_points() {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            py.run(
                c"class Bad(int):\n def __float__(self):\n  raise RuntimeError('bad conversion')",
                Some(&globals),
                None,
            )
            .unwrap();
            let record = PyDict::new(py);
            record.set_item("first", 1.0).unwrap();
            record
                .set_item(
                    "bad",
                    globals
                        .get_item("Bad")
                        .unwrap()
                        .unwrap()
                        .call1((2,))
                        .unwrap(),
                )
                .unwrap();
            record.set_item("after", 3.0).unwrap();
            let series = NativeMetricSeries::new(2);
            let error = series
                .add_validated_record(py, "0".into(), 1.5, &record)
                .unwrap_err();
            assert!(error.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            let state = series.state.lock();
            assert_eq!(state.key_cache.len(), 2);
            assert!(state.key_cache.contains_key("bad"));
            assert!(!state.key_cache.contains_key("after"));
            assert_eq!(
                export(&state.values[&ascii_key("first")][0]),
                ("0".into(), 1.5, 1.0)
            );
            assert!(!state.values.contains_key(&ascii_key("bad")));
        });
    }

    #[test]
    fn stored_samples_share_metadata_and_trim_without_growing() {
        let sample = Arc::new(SampleTime {
            step: "42".into(),
            time: 12.0,
        });
        let mut points = VecDeque::new();
        for value in 0..8192 {
            push_point(&mut points, 4096, (sample.clone(), value as f64));
        }
        assert!(Arc::ptr_eq(&points[0].0, &points[8191].0));
        let capacity = points.capacity();
        push_point(&mut points, 4096, (sample.clone(), 8192.0));
        assert_eq!(points.len(), 4096);
        assert_eq!(points.capacity(), capacity);
        assert_eq!(export(points.front().unwrap()), ("42".into(), 12.0, 4097.0));
        assert_eq!(export(points.back().unwrap()), ("42".into(), 12.0, 8192.0));
        #[cfg(target_pointer_width = "64")]
        assert_eq!(std::mem::size_of::<StoredPoint>(), 16);
    }
}
