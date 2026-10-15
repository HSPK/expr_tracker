use super::json::{JsonObject, Step};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::{BTreeMap, HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

static DATA_OPENS: AtomicUsize = AtomicUsize::new(0);

#[path = "reader/segments.rs"]
mod segments;
use segments::open_data;
pub use segments::{
    discover_runs, parse_row, segment_paths, visit_rows, visit_rows_from, HistoryQuery, RawRow,
    SegmentedReader,
};

#[pyclass]
struct NativeReader {
    path: PathBuf,
    snapshot: Option<SegmentedReader>,
    index: Option<Vec<(Step, usize, u64)>>,
    merge: Option<bool>,
    cache: Arc<Mutex<Option<(u64, Vec<(Step, usize, u64)>, bool)>>>,
    warnings: Arc<Mutex<Vec<String>>>,
    scanned: Arc<AtomicUsize>,
    integer_limit: usize,
}
impl NativeReader {
    fn reader(&self, end: Option<u64>) -> io::Result<SegmentedReader> {
        let mut reader = self
            .snapshot
            .clone()
            .map(Ok)
            .unwrap_or_else(|| SegmentedReader::open(&self.path))?
            .snapshot(end);
        reader.scanned = self.scanned.clone();
        reader.integer_limit = self.integer_limit;
        reader.warnings = Some(self.warnings.clone());
        Ok(reader)
    }
    fn prepared(&self, end: Option<u64>) -> io::Result<SegmentedReader> {
        let mut reader = self.reader(end)?;
        let cached = self
            .cache
            .lock()
            .clone()
            .filter(|(size, _, _)| *size == reader.limit());
        let (index, merge) = if let Some((_, index, merge)) = cached {
            (index, merge)
        } else {
            let meta = self.path.with_extension("meta.json");
            if let Err(error) = fs::read_to_string(&meta) {
                if error.kind() != io::ErrorKind::NotFound {
                    self.warnings.lock().push(format!(
                        "Could not read metadata from {}: {}; scanning history file.",
                        meta.display(),
                        error
                    ));
                }
            }
            let metadata = reader.metadata(1000)?;
            *self.cache.lock() = Some((reader.limit(), metadata.0.clone(), metadata.1));
            metadata
        };
        reader.hints = Some((
            self.index
                .clone()
                .unwrap_or(index)
                .into_iter()
                .filter(|e| e.2 < reader.limit())
                .collect(),
            self.merge.unwrap_or(merge),
        ));
        Ok(reader)
    }
}
#[pymethods]
impl NativeReader {
    #[new]
    #[pyo3(signature=(path,index=None,merge=None,max_digits=4300))]
    fn new(
        path: PathBuf,
        index: Option<Vec<(String, usize, u64)>>,
        merge: Option<bool>,
        max_digits: usize,
    ) -> Self {
        Self {
            path,
            snapshot: None,
            index: index.map(|v| v.into_iter().map(|(s, l, o)| (Step(s), l, o)).collect()),
            merge,
            cache: Arc::new(Mutex::new(None)),
            warnings: Arc::new(Mutex::new(Vec::new())),
            scanned: Arc::new(AtomicUsize::new(0)),
            integer_limit: max_digits,
        }
    }
    #[pyo3(signature=(end=None))]
    fn snapshot(&self, py: Python<'_>, end: Option<u64>) -> PyResult<Self> {
        let reader = py.detach(|| self.reader(end))?;
        Ok(Self {
            path: self.path.clone(),
            snapshot: Some(reader),
            index: self.index.clone(),
            merge: self.merge,
            cache: self.cache.clone(),
            warnings: Arc::new(Mutex::new(Vec::new())),
            scanned: self.scanned.clone(),
            integer_limit: self.integer_limit,
        })
    }
    #[getter]
    fn size(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(py.detach(|| self.reader(None))?.limit())
    }
    #[getter]
    fn merge(&self, py: Python<'_>) -> PyResult<bool> {
        if let Some(m) = self.merge {
            return Ok(m);
        }
        Ok(py.detach(|| self.prepared(None)?.metadata(1000))?.1)
    }
    #[getter]
    fn index(&self, py: Python<'_>) -> PyResult<Vec<(String, usize, u64)>> {
        let entries = if let Some(index) = &self.index {
            index.clone()
        } else {
            py.detach(|| self.prepared(None)?.metadata(1000))?.0
        };
        Ok(entries.into_iter().map(|(s, l, o)| (s.0, l, o)).collect())
    }
    #[pyo3(signature=(start=0,end=None))]
    fn scan(
        &self,
        py: Python<'_>,
        start: u64,
        end: Option<u64>,
    ) -> PyResult<Vec<(u64, Py<PyBytes>)>> {
        let rows = py.detach(|| {
            let reader = self.reader(end)?;
            let mut rows = Vec::new();
            reader.visit(start, true, |r| {
                rows.push(r);
                Ok(())
            })?;
            Ok::<_, io::Error>(rows)
        })?;
        Ok(rows
            .into_iter()
            .map(|r| (r.offset, PyBytes::new(py, &r.bytes).unbind()))
            .collect())
    }
    #[pyo3(signature=(count,end=None))]
    fn tail_raw(
        &self,
        py: Python<'_>,
        count: usize,
        end: Option<u64>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        let rows = py.detach(|| {
            let reader = self.reader(end)?;
            let mut rows = Vec::new();
            if count > 0 {
                reader.reverse(|r| {
                    rows.push(r.bytes);
                    Ok(rows.len() < count)
                })?;
            }
            rows.reverse();
            Ok::<_, io::Error>(rows)
        })?;
        Ok(rows
            .into_iter()
            .map(|r| PyBytes::new(py, &r).unbind())
            .collect())
    }
    #[pyo3(signature=(count=None,start=None,end=None,stop_offset=None))]
    fn query(
        &self,
        py: Python<'_>,
        count: Option<usize>,
        start: Option<String>,
        end: Option<String>,
        stop_offset: Option<u64>,
    ) -> PyResult<Vec<String>> {
        let query = HistoryQuery {
            limit: count,
            start: start.map(Step),
            end: end.map(Step),
            merge: self.merge,
        };
        Ok(py.detach(|| self.prepared(stop_offset)?.query_encoded(&query))?)
    }
    fn offset_of_step(&self, py: Python<'_>, step: String) -> PyResult<u64> {
        Ok(py.detach(|| self.prepared(None)?.offset_step(&Step(step)))?)
    }
    fn offset_of_line(&self, py: Python<'_>, line: usize) -> PyResult<u64> {
        Ok(py.detach(|| self.prepared(None)?.offset_line(line))?)
    }
    fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut *self.warnings.lock())
    }
    fn parse_rows(
        &self,
        py: Python<'_>,
        records: Vec<Vec<u8>>,
        max_digits: usize,
    ) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> io::Result<Vec<String>> {
            let reader = SegmentedReader {
                path: self.path.clone(),
                segments: Vec::new(),
                size: 0,
                limit: 0,
                hints: None,
                scanned: self.scanned.clone(),
                integer_limit: max_digits,
                warnings: Some(self.warnings.clone()),
            };
            Ok(records
                .into_iter()
                .filter_map(|bytes| reader.parse_row(&bytes).map(|row| row.encode()))
                .collect())
        })?)
    }
    #[getter]
    fn scanned_rows(&self) -> usize {
        self.scanned.load(Ordering::Relaxed)
    }
    fn clear_stats(&self) {
        self.scanned.store(0, Ordering::Relaxed);
    }
    #[pyo3(signature=(start=0,end=None))]
    fn iter_raw(
        &self,
        py: Python<'_>,
        start: u64,
        end: Option<u64>,
    ) -> PyResult<NativeRawIterator> {
        let reader = py.detach(|| self.reader(end))?;
        Ok(NativeRawIterator {
            cursor: Mutex::new(RawCursor {
                reader,
                start,
                index: 0,
                base: 0,
                offset: 0,
                physical_length: 0,
                input: None,
            }),
        })
    }
    fn count_lines(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(py.detach(|| -> io::Result<usize> {
            let mut count = 0;
            self.reader(None)?.visit(0, true, |_| {
                count += 1;
                Ok(())
            })?;
            Ok(count)
        })?)
    }
    #[pyo3(signature=(count,end=None,newer=Vec::new()))]
    fn select_recent(
        &self,
        py: Python<'_>,
        count: usize,
        end: Option<u64>,
        newer: Vec<String>,
    ) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> io::Result<Vec<String>> {
            let reader = self.reader(end)?;
            let mut selected = HashSet::new();
            if count == 0 {
                return Ok(Vec::new());
            }
            for step in newer {
                selected.insert(Step(step));
                if selected.len() == count {
                    break;
                }
            }
            if selected.len() < count {
                reader.reverse(|row| {
                    if let Some(record) = reader.parse_row(&row.bytes) {
                        selected.insert(record.step().unwrap());
                    }
                    Ok(selected.len() < count)
                })?;
            }
            Ok(selected.into_iter().map(|s| s.0).collect())
        })?)
    }
    #[pyo3(signature=(selected,end=None))]
    fn selected(
        &self,
        py: Python<'_>,
        selected: Vec<String>,
        end: Option<u64>,
    ) -> PyResult<Vec<String>> {
        Ok(py.detach(|| -> io::Result<Vec<String>> {
            let selected: HashSet<_> = selected.into_iter().map(Step).collect();
            let mut merged = BTreeMap::<Step, JsonObject>::new();
            if !selected.is_empty() {
                let reader = self.reader(end)?;
                reader.visit(0, true, |row| {
                    if let Some(record) = reader.parse_row(&row.bytes) {
                        let step = record.step().unwrap();
                        if selected.contains(&step) {
                            merged.entry(step).or_default().merge(&record);
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(merged.into_values().map(|r| r.encode()).collect())
        })?)
    }
}
struct RawCursor {
    reader: SegmentedReader,
    start: u64,
    index: usize,
    base: u64,
    offset: u64,
    physical_length: u64,
    input: Option<BufReader<std::io::Take<File>>>,
}
impl RawCursor {
    fn next(&mut self) -> io::Result<Option<Vec<u8>>> {
        loop {
            if let Some(input) = &mut self.input {
                let mut row = Vec::new();
                let count = input.read_until(b'\n', &mut row)?;
                if count == 0 {
                    self.input = None;
                    self.base += self.physical_length;
                    continue;
                }
                self.offset += count as u64;
                if row.last() != Some(&b'\n')
                    && self.base + self.physical_length > self.reader.limit
                {
                    self.input = None;
                    self.base += self.physical_length;
                    continue;
                }
                if row.iter().any(|b| !b.is_ascii_whitespace()) {
                    self.reader.scanned.fetch_add(1, Ordering::Relaxed);
                    return Ok(Some(row));
                }
                continue;
            }
            let Some((path, length)) = self.reader.segments.get(self.index) else {
                return Ok(None);
            };
            self.index += 1;
            if self.base >= self.reader.limit {
                return Ok(None);
            }
            let end = (*length).min(self.reader.limit - self.base);
            let start = self.start.saturating_sub(self.base);
            if start >= end {
                self.base += length;
                continue;
            }
            let mut file = open_data(path)?;
            file.seek(SeekFrom::Start(start))?;
            self.input = Some(BufReader::with_capacity(256 * 1024, file.take(end - start)));
            self.offset = start;
            self.physical_length = *length;
        }
    }
}
#[pyclass]
struct NativeRawIterator {
    cursor: Mutex<RawCursor>,
}
#[pymethods]
impl NativeRawIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        Ok(py
            .detach(|| self.cursor.lock().next())?
            .map(|row| PyBytes::new(py, &row).unbind()))
    }
}
#[pyfunction]
fn history_recent_steps(steps: &Bound<'_, PyAny>, count: usize) -> PyResult<Vec<String>> {
    let mut selected = HashSet::new();
    if count == 0 {
        return Ok(Vec::new());
    }
    for step in steps.try_iter()? {
        let step = step?;
        let value = step.str()?.to_string_lossy().into_owned();
        selected.insert(if value == "True" {
            "1".into()
        } else if value == "False" {
            "0".into()
        } else {
            value
        });
        if selected.len() == count {
            break;
        }
    }
    Ok(selected.into_iter().collect())
}
#[pyfunction]
fn history_merge(
    py: Python<'_>,
    records: Vec<String>,
    selected: Option<Vec<String>>,
) -> Vec<String> {
    py.detach(|| {
        let selected = selected.map(|v| v.into_iter().map(Step).collect::<HashSet<_>>());
        let mut merged = BTreeMap::<Step, JsonObject>::new();
        for text in records {
            if let Some(record) = JsonObject::parse(&text) {
                if let Some(step) = record.step() {
                    if selected.as_ref().is_none_or(|s| s.contains(&step)) {
                        merged.entry(step).or_default().merge(&record);
                    }
                }
            }
        }
        merged.into_values().map(|r| r.encode()).collect()
    })
}
#[pyfunction]
#[pyo3(signature=(records,max_digits=4300))]
fn history_parse(py: Python<'_>, records: Vec<Vec<u8>>, max_digits: usize) -> Vec<String> {
    py.detach(|| {
        records
            .into_iter()
            .filter_map(|bytes| {
                if super::json::exceeds_integer_limit(&bytes, max_digits) {
                    None
                } else {
                    parse_row(&bytes).map(|r| r.encode())
                }
            })
            .collect()
    })
}
#[pyfunction]
fn history_io_count() -> usize {
    DATA_OPENS.load(Ordering::Relaxed)
}
#[pyfunction]
fn history_segments(py: Python<'_>, path: PathBuf) -> PyResult<Vec<PathBuf>> {
    Ok(py.detach(|| segment_paths(&path))?)
}
#[pyfunction]
fn history_atomic_write(py: Python<'_>, path: PathBuf, contents: Vec<u8>) -> PyResult<()> {
    use std::io::Write;
    Ok(py.detach(|| -> io::Result<()> {
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        file.write_all(&contents)?;
        file.persist(path).map_err(|error| error.error)?;
        Ok(())
    })?)
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeReader>()?;
    module.add_function(wrap_pyfunction!(history_merge, module)?)?;
    module.add_class::<NativeRawIterator>()?;
    module.add_function(wrap_pyfunction!(history_parse, module)?)?;
    module.add_function(wrap_pyfunction!(history_io_count, module)?)?;
    module.add_function(wrap_pyfunction!(history_segments, module)?)?;
    module.add_function(wrap_pyfunction!(history_atomic_write, module)?)?;
    module.add_function(wrap_pyfunction!(history_recent_steps, module)?)?;
    Ok(())
}
