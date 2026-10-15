use super::{BTreeMap, BTreeSet, BinaryHeap, Ordering, PyValueError, Reverse};
use pyo3::prelude::*;

#[derive(Copy, Clone, PartialEq)]
pub(super) struct Number(pub(super) f64);
impl Eq for Number {}
impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Number {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or_else(|| self.0.total_cmp(&other.0))
    }
}

// Decimal depth strings avoid limiting valid Python integer depths to u64.
pub(super) fn decrement(depth: &str) -> String {
    if depth == "0" {
        return "-1".into();
    }
    let mut bytes = depth.as_bytes().to_vec();
    for c in bytes.iter_mut().rev() {
        if *c > b'0' {
            *c -= 1;
            break;
        }
        *c = b'9';
    }
    let value = String::from_utf8(bytes).unwrap();
    let value = value.trim_start_matches('0');
    if value.is_empty() {
        "0".into()
    } else {
        value.to_owned()
    }
}

#[pyclass(name = "NativeLanePool")]
#[derive(Default)]
pub struct LanePool {
    pub(super) stacks: Vec<Vec<(Number, Option<String>)>>,
    pub(super) ends: BinaryHeap<Reverse<(Number, usize)>>,
    pub(super) free: BTreeSet<usize>,
    pub(super) available: BTreeMap<Option<String>, BTreeSet<(Reverse<Number>, usize)>>,
}

impl LanePool {
    pub(super) fn remove_top(&mut self, lane: usize) {
        let (end, depth) = self.stacks[lane].last().unwrap();
        let heap = self.available.get_mut(depth).unwrap();
        heap.remove(&(Reverse(*end), lane));
        if heap.is_empty() {
            self.available.remove(depth);
        }
    }
    pub(super) fn add_top(&mut self, lane: usize) {
        let (end, depth) = self.stacks[lane].last().unwrap();
        self.available
            .entry(depth.clone())
            .or_default()
            .insert((Reverse(*end), lane));
    }
    pub(super) fn place_inner(
        &mut self,
        start: f64,
        duration: f64,
        depth: Option<String>,
    ) -> PyResult<usize> {
        let end = Number(start + duration * 1000.0);
        if !end.0.is_finite() {
            return Err(PyValueError::new_err(
                "selected span timestamps exceed finite trace range",
            ));
        }
        while self.ends.peek().is_some_and(|v| v.0 .0 .0 <= start) {
            let Reverse((_, lane)) = self.ends.pop().unwrap();
            if self.stacks[lane].last().is_none_or(|v| v.0 .0 > start) {
                continue;
            }
            self.remove_top(lane);
            while self.stacks[lane].last().is_some_and(|v| v.0 .0 <= start) {
                self.stacks[lane].pop();
            }
            if self.stacks[lane].is_empty() {
                self.free.insert(lane);
            } else {
                self.add_top(lane);
            }
        }
        let parent_depth = depth.as_deref().map(decrement);
        let parent = self
            .available
            .get(&parent_depth)
            .and_then(|v| v.first())
            .filter(|v| v.0 .0 >= end)
            .map(|v| v.1);
        let free = self.free.first().copied();
        let lane = if free.is_some_and(|v| parent.is_none_or(|p| v < p)) {
            let lane = free.unwrap();
            self.free.remove(&lane);
            lane
        } else if let Some(lane) = parent {
            self.remove_top(lane);
            lane
        } else {
            self.stacks.push(vec![]);
            self.stacks.len() - 1
        };
        self.stacks[lane].push((end, depth));
        self.add_top(lane);
        self.ends.push(Reverse((end, lane)));
        Ok(lane)
    }
}

#[pymethods]
impl LanePool {
    #[new]
    pub(super) fn new() -> Self {
        Self::default()
    }
    pub(super) fn place(
        &mut self,
        start: f64,
        duration: f64,
        depth: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<usize> {
        self.place_inner(
            start,
            duration,
            depth
                .filter(|v| !v.is_none())
                .map(|v| v.str()?.extract::<String>())
                .transpose()?,
        )
    }
    #[getter]
    pub(super) fn stacks(&self) -> Vec<Vec<(f64, Option<String>)>> {
        self.stacks
            .iter()
            .map(|s| s.iter().map(|v| (v.0 .0, v.1.clone())).collect())
            .collect()
    }
    #[getter]
    pub(super) fn ends(&self) -> Vec<(f64, usize)> {
        self.ends.iter().map(|v| (v.0 .0 .0, v.0 .1)).collect()
    }
    #[getter]
    pub(super) fn available(&self) -> BTreeMap<Option<String>, Vec<(f64, usize)>> {
        self.available
            .iter()
            .map(|(d, heap)| (d.clone(), heap.iter().map(|v| (v.0 .0 .0, v.1)).collect()))
            .collect()
    }
}
