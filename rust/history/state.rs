use super::json::{add_int, float_json, number, JsonObject, Step};
use super::reader::{HistoryQuery, SegmentedReader};
use super::record::{normalize_numpy_scalars, scalar_object, NativeRecord, SharedLine};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

#[derive(Clone)]
struct Cached {
    row: i64,
    step: Step,
    line: SharedLine,
}
struct State {
    open: Option<Step>,
    data: JsonObject,
    weights: HashMap<String, f64>,
    time: f64,
    next: Step,
    last: Option<Step>,
    emitted: Option<Step>,
    row: i64,
    cache: VecDeque<Cached>,
    bytes: usize,
    cache_bytes: usize,
    cache_rows: usize,
    disk_prefix: bool,
    merge: bool,
    evicted: usize,
    queries: usize,
    disk_queries: usize,
    last_commit: Option<f64>,
    allow: bool,
    json_keys: HashMap<String, Vec<u8>>,
}
impl State {
    fn current(&self) -> Step {
        self.open.clone().unwrap_or_else(|| self.next.clone())
    }
    fn opened(&self) -> Option<JsonObject> {
        let step = self.open.as_ref()?;
        let mut record = JsonObject(
            [
                ("_step".into(), step.0.clone()),
                ("_time".into(), float_json(self.time)),
            ]
            .into_iter()
            .collect(),
        );
        record.merge(&self.data);
        Some(record)
    }
    fn close(&mut self) -> Option<JsonObject> {
        let step = self.open.take()?;
        let mut record = JsonObject(
            [
                ("_step".into(), step.0.clone()),
                ("_time".into(), float_json(self.time)),
            ]
            .into_iter()
            .collect(),
        );
        record.0.extend(std::mem::take(&mut self.data.0));
        self.last = Some(step.clone());
        self.next = step.next();
        self.data = JsonObject::default();
        self.weights.clear();
        Some(record)
    }
    fn over(&self) -> bool {
        self.bytes > self.cache_bytes || self.cache.len() > self.cache_rows
    }
}
#[pyclass]
pub struct NativeHistoryState {
    state: Mutex<State>,
    owner: u32,
}
#[pymethods]
impl NativeHistoryState {
    #[new]
    fn new(cache_bytes: usize, cache_rows: usize, allow: bool) -> Self {
        Self {
            state: Mutex::new(State {
                open: None,
                data: JsonObject::default(),
                weights: HashMap::new(),
                time: 0.0,
                next: Step("0".into()),
                last: None,
                emitted: None,
                row: 0,
                cache: VecDeque::new(),
                bytes: 0,
                cache_bytes,
                cache_rows: cache_rows.max(1),
                disk_prefix: false,
                merge: false,
                evicted: 0,
                queries: 0,
                disk_queries: 0,
                last_commit: None,
                allow,
                json_keys: HashMap::new(),
            }),
            owner: std::process::id(),
        }
    }
    fn resume(&self, rows: i64, max_step: Option<String>, merge: bool) {
        let mut s = self.state.lock();
        s.row = rows;
        s.disk_prefix = rows > 0;
        s.merge = merge;
        if let Some(step) = max_step.map(Step) {
            s.next = step.next();
            s.last = Some(step.clone());
            s.emitted = Some(step);
        }
    }
    fn ingest(
        &self,
        py: Python<'_>,
        encoded: String,
        step: Option<String>,
        commit: bool,
        accumulate: bool,
        reductions: HashMap<String, String>,
        weight: f64,
        now: f64,
    ) -> PyResult<(Option<String>, Vec<String>, Vec<(String, String)>)> {
        if self.owner != std::process::id() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tracker objects cannot be reused after fork.",
            ));
        }
        let encoded = JsonObject::parse(&encoded).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Metrics must be an encoded JSON object")
        })?;
        let (resolved, closed, warnings) = self.ingest_object(
            py, encoded, step, commit, accumulate, reductions, weight, now,
        )?;
        Ok((
            resolved,
            closed.into_iter().map(|record| record.encode()).collect(),
            warnings,
        ))
    }
    fn ingest_values(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
        step: Option<String>,
        commit: bool,
        accumulate: bool,
        reductions: HashMap<String, String>,
        weight: f64,
        now: f64,
    ) -> PyResult<Option<(Option<String>, Vec<NativeRecord>, Vec<(String, String)>)>> {
        if self.owner != std::process::id() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Tracker objects cannot be reused after fork.",
            ));
        }
        let snapshot = values.copy()?;
        normalize_numpy_scalars(py, &snapshot)?;
        {
            let mut s = self.state.lock();
            if s.open.is_none() && commit && !accumulate {
                let target = step
                    .as_ref()
                    .map(|value| Step(value.clone()))
                    .unwrap_or_else(|| s.current());
                let Some((record, warnings)) = NativeRecord::from_values(
                    py,
                    &snapshot,
                    target.clone(),
                    now,
                    &mut s.json_keys,
                )?
                else {
                    return Ok(None);
                };
                if step.is_some() && !s.allow && s.last.as_ref().is_some_and(|last| target < *last)
                {
                    return Ok(Some((None, Vec::new(), vec![("".into(), format!(
                        "Dropping metrics for step {}: step must be >= {} (set step_policy=\"allow\" to keep out-of-order writes).",
                        target.0, s.last.as_ref().unwrap().0
                    ))])));
                }
                s.last = Some(target.clone());
                s.next = target.next();
                s.time = now;
                s.data = JsonObject::default();
                s.weights.clear();
                return Ok(Some((Some(target.0), vec![record], warnings)));
            }
        }
        let Some(encoded) = scalar_object(&snapshot)? else {
            return Ok(None);
        };
        let fresh = self.state.lock().open.is_none() && commit && !accumulate;
        let (resolved, closed, warnings) = self.ingest_object(
            py, encoded, step, commit, accumulate, reductions, weight, now,
        )?;
        let closed = closed
            .into_iter()
            .map(|record| {
                let reuse = !accumulate && record.get("_step") == resolved.as_deref();
                NativeRecord::new(py, record, &snapshot, reuse, fresh && reuse)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Some((resolved, closed, warnings)))
    }
}
impl NativeHistoryState {
    fn ingest_object(
        &self,
        py: Python<'_>,
        encoded: JsonObject,
        step: Option<String>,
        commit: bool,
        accumulate: bool,
        reductions: HashMap<String, String>,
        weight: f64,
        now: f64,
    ) -> PyResult<(Option<String>, Vec<JsonObject>, Vec<(String, String)>)> {
        // Tiny GIL handoffs let CPU-bound readers repeatedly preempt producers.
        let substantial = encoded
            .0
            .iter()
            .map(|(key, value)| key.len() + value.len())
            .sum::<usize>()
            > 64 * 1024
            || self
                .state
                .lock()
                .data
                .0
                .iter()
                .map(|(key, value)| key.len() + value.len())
                .sum::<usize>()
                > 64 * 1024;
        let ingest = || {
            let mut s = self.state.lock();
            let mut closed = Vec::new();
            let mut warnings = Vec::new();
            if let Some(step) = step.as_ref().map(|s| Step(s.clone())) {
                let reference = s.open.as_ref().or(s.last.as_ref());
                if !s.allow && reference.is_some_and(|r| step < *r) {
                    warnings.push(("".into(),format!("Dropping metrics for step {}: step must be >= {} (set step_policy=\"allow\" to keep out-of-order writes).",step.0,reference.unwrap().0)));
                    return (None, closed, warnings);
                }
            }
            let target = step.map(Step).unwrap_or_else(|| s.current());
            if s.open.as_ref() != Some(&target) {
                if let Some(record) = s.close() {
                    closed.push(record);
                }
                s.open = Some(target.clone());
                s.time = now;
                s.data = JsonObject::default();
                s.weights.clear();
            }
            for (key, value) in encoded.0 {
                if key == "_step" || key == "_time" {
                    warnings.push((
                        key.clone(),
                        format!("Metric name {key:?} is reserved and will be ignored."),
                    ));
                    continue;
                }
                if !accumulate {
                    s.weights.remove(&key);
                    if s.data.get(&key).is_some() {
                        warnings.push((key.clone(),format!("Metric {key:?} logged twice for step {}; keeping the latest value.",target.0)));
                    }
                    s.data.set(key, value);
                    continue;
                }
                let previous = s.data.get(&key).map(str::to_owned);
                let reduction = reductions.get(&key).map(String::as_str);
                let numeric = number(&value);
                let current = previous.as_deref().and_then(number);
                let merged = if let Some(value_num) = numeric {
                    if reduction == Some("mean") {
                        let previous_weight = if current.is_some() {
                            s.weights.get(&key).copied().unwrap_or(0.0)
                        } else {
                            0.0
                        };
                        s.weights.insert(key.clone(), previous_weight + weight);
                        if previous_weight == 0.0 {
                            value
                        } else if weight == 0.0 {
                            previous.unwrap()
                        } else {
                            float_json(
                                (current.unwrap() * previous_weight + value_num * weight)
                                    / (previous_weight + weight),
                            )
                        }
                    } else {
                        s.weights.remove(&key);
                        if let Some(current_num) = current {
                            if reduction == Some("max") {
                                let exact = Step::parse(&value)
                                    .zip(previous.as_deref().and_then(Step::parse));
                                if value_num.is_nan()
                                    || exact
                                        .as_ref()
                                        .map_or(value_num > current_num, |(value, current)| {
                                            value > current
                                        })
                                {
                                    value
                                } else {
                                    previous.unwrap()
                                }
                            } else if Step::parse(&value).is_some()
                                && previous.as_deref().and_then(Step::parse).is_some()
                            {
                                add_int(&value, &previous.unwrap())
                            } else {
                                float_json(value_num + current_num)
                            }
                        } else {
                            value
                        }
                    }
                } else {
                    s.weights.remove(&key);
                    value
                };
                s.data.set(key, merged);
            }
            if commit {
                if let Some(record) = s.close() {
                    closed.push(record);
                }
            }
            (Some(target.0), closed, warnings)
        };
        Ok(if substantial {
            py.detach(ingest)
        } else {
            ingest()
        })
    }
}

#[path = "state/operations.rs"]
mod operations;

fn in_range(step: &Step, start: &Option<Step>, end: &Option<Step>) -> bool {
    start.as_ref().is_none_or(|s| step >= s) && end.as_ref().is_none_or(|s| step < s)
}
#[pyfunction]
fn history_scan_tail(steps: Vec<String>, count: usize) -> (Vec<usize>, bool) {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    let mut complete = false;
    for (index, step) in steps.into_iter().enumerate().rev() {
        if !seen.contains(&step) && seen.len() == count {
            complete = true;
            break;
        }
        seen.insert(step);
        result.push(index);
    }
    result.reverse();
    (result, complete)
}
#[pyfunction]
fn history_scan_range(
    steps: Vec<String>,
    start: Option<String>,
    end: Option<String>,
    forward: bool,
) -> Vec<usize> {
    let start = start.map(Step);
    let end = end.map(Step);
    let steps: Vec<_> = steps.into_iter().map(Step).collect();
    let mut indices: Vec<_> = (0..steps.len()).collect();
    if !forward {
        indices.reverse();
    }
    let mut result = Vec::new();
    for index in indices {
        let step = &steps[index];
        if (forward && end.as_ref().is_some_and(|end| step >= end))
            || (!forward && start.as_ref().is_some_and(|start| step < start))
        {
            break;
        }
        if in_range(step, &start, &end) {
            result.push(index);
        }
    }
    if !forward {
        result.reverse();
    }
    result
}
#[pyfunction]
fn history_nearer_front(
    start: Option<String>,
    end: Option<String>,
    oldest: String,
    newest: String,
) -> bool {
    let front = add_int(start.as_deref().unwrap_or(&oldest), &negate(&oldest));
    let back = add_int(
        &newest,
        &negate(&end.map_or_else(|| newest.clone(), |end| add_int(&end, "-1"))),
    );
    Step(front) <= Step(back)
}
fn negate(value: &str) -> String {
    if value == "0" {
        "0".into()
    } else if let Some(digits) = value.strip_prefix('-') {
        digits.into()
    } else {
        format!("-{value}")
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeHistoryState>()?;
    module.add_function(wrap_pyfunction!(history_scan_tail, module)?)?;
    module.add_function(wrap_pyfunction!(history_scan_range, module)?)?;
    module.add_function(wrap_pyfunction!(history_nearer_front, module)?)?;
    Ok(())
}
