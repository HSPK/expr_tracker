use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString};
use rusqlite::{params, Connection};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

mod lanes;
use lanes::*;

fn number(value: &Bound<'_, PyAny>) -> PyResult<f64> {
    if value.is_instance_of::<PyBool>()
        || !(value.is_instance_of::<PyInt>()
            || value.is_instance_of::<PyFloat>()
            || value.is_instance_of::<PyString>())
    {
        return Err(PyValueError::new_err("not a numeric timestamp"));
    }
    let number: f64 =
        if value.is_exact_instance_of::<PyInt>() || value.is_exact_instance_of::<PyFloat>() {
            value.extract()?
        } else {
            value
                .call_method0("__float__")
                .or_else(|_| {
                    value
                        .py()
                        .import("builtins")?
                        .getattr("float")?
                        .call1((value,))
                })?
                .extract()?
        };
    if !number.is_finite() {
        return Err(PyValueError::new_err("not a finite timestamp"));
    }
    Ok(number)
}

fn get<'py>(record: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(record
        .get_item(key)?
        .unwrap_or_else(|| record.py().None().into_bound(record.py())))
}
fn depth(record: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
    let value = get(record, "depth")?;
    if value.is_none() {
        return Ok(None);
    }
    if value.is_instance_of::<PyBool>() || !value.is_instance_of::<PyInt>() || value.lt(0)? {
        return Err(PyValueError::new_err("not a valid span depth"));
    }
    Ok(Some(value.str()?.extract()?))
}

struct Filters {
    steps: Option<(Option<Py<PyAny>>, Option<Py<PyAny>>)>,
    times: Option<(Option<f64>, Option<f64>)>,
    empty: bool,
}
impl Filters {
    fn new(
        py: Python<'_>,
        steps: Option<&Bound<'_, PyAny>>,
        times: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut empty = false;
        let steps = steps
            .filter(|v| !v.is_none())
            .map(|v| -> PyResult<_> {
                let low = v.get_item(0)?;
                let high = v.get_item(1)?;
                if !low.is_none() && !high.is_none() && low.ge(&high)? {
                    empty = true;
                }
                Ok((
                    if low.is_none() {
                        None
                    } else {
                        Some(low.unbind())
                    },
                    if high.is_none() {
                        None
                    } else {
                        Some(high.unbind())
                    },
                ))
            })
            .transpose()?;
        let times = times
            .filter(|v| !v.is_none())
            .map(|v| -> PyResult<_> {
                if v.len()? != 2 {
                    return Err(PyValueError::new_err(
                        "time_range must contain two epoch-second bounds",
                    ));
                }
                let mut bounds = vec![];
                for i in 0..2 {
                    let item = v.get_item(i)?;
                    if item.is_none() {
                        bounds.push(None);
                    } else if item.is_instance_of::<PyString>() {
                        return Err(PyValueError::new_err(
                            "time_range bounds must be finite epoch seconds or None",
                        ));
                    } else {
                        bounds.push(Some(number(&item).map_err(|_| {
                            PyValueError::new_err(
                                "time_range bounds must be finite epoch seconds or None",
                            )
                        })?));
                    }
                }
                if let (Some(low), Some(high)) = (bounds[0], bounds[1]) {
                    empty |= low >= high;
                }
                Ok((bounds[0], bounds[1]))
            })
            .transpose()?;
        let _ = py;
        Ok(Self {
            steps,
            times,
            empty,
        })
    }

    fn accepts(&self, py: Python<'_>, record: &Bound<'_, PyDict>) -> PyResult<bool> {
        if self.empty {
            return Ok(false);
        }
        let start = number(&get(record, "start")?)?;
        let duration = number(&get(record, "dur_ms")?)?;
        let end = start + duration / 1000.0;
        if duration < 0.0
            || !end.is_finite()
            || !(start * 1_000_000.0).is_finite()
            || !(duration * 1000.0).is_finite()
        {
            return Ok(false);
        }
        if let Some((low, high)) = self.times {
            if duration == 0.0 || low.is_some_and(|v| end <= v) || high.is_some_and(|v| start >= v)
            {
                return Ok(false);
            }
        }
        if let Some((low, high)) = &self.steps {
            let step = record
                .get_item("_step")?
                .unwrap_or_else(|| 0_i64.into_pyobject(py).unwrap().into_any());
            let step = if step.is_instance_of::<PyInt>() && !step.is_instance_of::<PyBool>() {
                step
            } else {
                number(&step)?.into_pyobject(py)?.into_any()
            };
            if low
                .as_ref()
                .is_some_and(|v| step.lt(v.bind(py)).unwrap_or(true))
                || high
                    .as_ref()
                    .is_some_and(|v| step.ge(v.bind(py)).unwrap_or(true))
            {
                return Ok(false);
            }
        }
        for key in ["args", "metrics"] {
            let value = get(record, key)?;
            if !value.is_none() && !value.is_instance_of::<PyDict>() {
                return Ok(false);
            }
        }
        let track = record
            .get_item("track")?
            .unwrap_or_else(|| 0_i64.into_pyobject(py).unwrap().into_any());
        if !(track.is_instance_of::<PyInt>()
            || track.is_instance_of::<PyFloat>()
            || track.is_instance_of::<PyString>())
        {
            return Ok(false);
        }
        if track.is_instance_of::<PyFloat>() && !track.extract::<f64>()?.is_finite() {
            return Ok(false);
        }
        depth(record)?;
        Ok(true)
    }
}

fn snapshot(path: &Path) -> std::io::Result<Vec<(File, u64)>> {
    let mut files = vec![path.to_owned()];
    let parts = path.with_file_name(format!(
        "{}.parts",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    if parts.is_dir() {
        let mut segments = std::fs::read_dir(parts)?
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|e| e == "jsonl")
                    && p.file_stem()
                        .is_some_and(|s| s.to_string_lossy().bytes().all(|c| c.is_ascii_digit()))
            })
            .collect::<Vec<_>>();
        segments.sort_by(|a, b| {
            let a = a.file_stem().unwrap().to_string_lossy();
            let b = b.file_stem().unwrap().to_string_lossy();
            let a = a.trim_start_matches('0');
            let b = b.trim_start_matches('0');
            a.len().cmp(&b.len()).then(a.cmp(b))
        });
        files.extend(segments);
    }
    let mut captured = vec![];
    for path in files {
        match File::open(path) {
            Ok(file) => {
                let size = file.metadata()?.len();
                captured.push((file, size));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(captured)
}

fn scan(
    py: Python<'_>,
    path: &Path,
    filters: &Filters,
    mut consume: impl FnMut(&Bound<'_, PyDict>, &[u8]) -> PyResult<()>,
) -> PyResult<()> {
    let loads = py.import("json")?.getattr("loads")?;
    for (file, size) in snapshot(path)? {
        let mut reader = BufReader::new(file.take(size));
        let mut raw = vec![];
        loop {
            raw.clear();
            if reader.read_until(b'\n', &mut raw)? == 0 {
                break;
            }
            let Ok(record) = loads.call1((PyBytes::new(py, &raw),)) else {
                continue;
            };
            let Ok(record) = record.cast::<PyDict>() else {
                continue;
            };
            if !filters.accepts(py, record).unwrap_or(false) {
                continue;
            }
            consume(record, &raw)?;
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(path, step_range=None, time_range=None))]
fn trace_read_spans(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    step_range: Option<&Bound<'_, PyAny>>,
    time_range: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyList>> {
    let path = crate::artifacts::path_value(path, false)?;
    let filters = Filters::new(py, step_range, time_range)?;
    let out = PyList::empty(py);
    scan(py, &path, &filters, |record, _raw| out.append(record))?;
    Ok(out.unbind())
}

fn relative(start: f64, origin: f64) -> PyResult<f64> {
    let value = start * 1_000_000.0 - origin * 1_000_000.0;
    if !value.is_finite() {
        return Err(PyValueError::new_err(
            "selected span timestamps exceed finite trace range",
        ));
    }
    Ok(value)
}

#[pyfunction]
fn trace_assign_lanes(py: Python<'_>, records: &Bound<'_, PyList>) -> PyResult<Py<PyDict>> {
    let tracks = PyDict::new(py);
    let mut grouped: Vec<Vec<(usize, f64, f64, Option<String>)>> = vec![];
    let mut origin = f64::INFINITY;
    for (index, record) in records.iter().enumerate() {
        let record = record.cast::<PyDict>()?;
        let track = record
            .get_item("track")?
            .unwrap_or_else(|| 0_i64.into_pyobject(py).unwrap().into_any());
        let track_id: usize = match tracks.get_item(&track)? {
            Some(id) => id.extract()?,
            None => {
                let id = grouped.len();
                tracks.set_item(track, id)?;
                grouped.push(vec![]);
                id
            }
        };
        let start = number(&get(record, "start")?)?;
        origin = origin.min(start);
        grouped[track_id].push((
            index,
            start,
            number(&get(record, "dur_ms")?)?,
            depth(record)?,
        ));
    }
    let result = PyDict::new(py);
    let mut offset = 0;
    for track in sorted_tracks(py, &tracks)? {
        let id: usize = tracks.get_item(track)?.unwrap().extract()?;
        let group = &mut grouped[id];
        group.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or_else(|| a.1.total_cmp(&b.1))
                .then(b.2.partial_cmp(&a.2).unwrap_or_else(|| b.2.total_cmp(&a.2)))
                .then_with(|| compare_depth(&a.3, &b.3))
                .then(a.0.cmp(&b.0))
        });
        let mut pool = LanePool::default();
        for (index, start, duration, depth) in group {
            result.set_item(
                *index,
                offset + pool.place_inner(relative(*start, origin)?, *duration, depth.clone())?,
            )?;
        }
        offset += pool.stacks.len();
    }
    Ok(result.unbind())
}

fn compare_depth(a: &Option<String>, b: &Option<String>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, _) => Ordering::Less,
        (_, None) => Ordering::Greater,
        (Some(a), Some(b)) => a.len().cmp(&b.len()).then(a.cmp(b)),
    }
}
fn sorted_tracks<'py>(
    _py: Python<'py>,
    tracks: &Bound<'py, PyDict>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let mut values = tracks
        .keys()
        .iter()
        .map(|track| {
            let key = track
                .str()?
                .call_method1("encode", ("utf-8", "surrogatepass"))?;
            Ok((key.cast::<PyBytes>()?.as_bytes().to_vec(), track))
        })
        .collect::<PyResult<Vec<_>>>()?;
    values.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(values.into_iter().map(|(_, track)| track).collect())
}
fn sql_error(error: rusqlite::Error) -> PyErr {
    PyOSError::new_err(error.to_string())
}

#[pyfunction]
fn diagnostic_float(py: Python<'_>, token: &str) -> PyResult<Py<PyAny>> {
    let value = token
        .parse::<f64>()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    if value.is_finite() {
        Ok(value.into_pyobject(py)?.into_any().unbind())
    } else {
        Ok(PyString::new(
            py,
            if value.is_sign_negative() {
                "-Infinity"
            } else {
                "Infinity"
            },
        )
        .into_any()
        .unbind())
    }
}

fn metadata<'py>(
    py: Python<'py>,
    kind: &str,
    pid: usize,
    tid: usize,
    label: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let args = PyDict::new(py);
    args.set_item("name", label)?;
    let record = PyDict::new(py);
    record.set_item("name", kind)?;
    record.set_item("ph", "M")?;
    record.set_item("pid", pid)?;
    record.set_item("tid", tid)?;
    record.set_item("args", args)?;
    Ok(record)
}

fn export(
    py: Python<'_>,
    files: Vec<(PathBuf, String)>,
    filters: &Filters,
    directory: &Path,
    observer: Option<&Bound<'_, PyAny>>,
    mut emit: impl FnMut(&Bound<'_, PyDict>) -> PyResult<()>,
) -> PyResult<usize> {
    let scratch = tempfile::Builder::new()
        .prefix(".expr-trace-")
        .tempdir_in(directory)?;
    if let Some(observer) = observer {
        observer.call1(("spool", scratch.path().join("spans.db")))?;
    }
    let db = Connection::open(scratch.path().join("spans.db")).map_err(sql_error)?;
    db.execute_batch("PRAGMA temp_store=FILE; PRAGMA cache_size=-2048; PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE spans(id INTEGER PRIMARY KEY, pid INTEGER, track INTEGER, start REAL, duration REAL, depth TEXT, record BLOB, lane INTEGER);").map_err(sql_error)?;
    let mut tracks = vec![];
    let mut origin = f64::INFINITY;
    db.execute_batch("BEGIN").map_err(sql_error)?;
    {
        let mut insert = db
            .prepare("INSERT INTO spans VALUES(NULL,?,?,?,?,?,?,NULL)")
            .map_err(sql_error)?;
        for (pid, (path, _)) in files.iter().enumerate() {
            let mapping = PyDict::new(py);
            scan(py, path, filters, |record, raw| {
                let start = number(&get(record, "start")?)?;
                origin = origin.min(start);
                let track = record
                    .get_item("track")?
                    .unwrap_or_else(|| 0_i64.into_pyobject(py).unwrap().into_any());
                let id: usize = match mapping.get_item(&track)? {
                    Some(v) => v.extract()?,
                    None => {
                        let id = mapping.len();
                        mapping.set_item(track, id)?;
                        id
                    }
                };
                insert
                    .execute(params![
                        pid as i64,
                        id as i64,
                        start,
                        number(&get(record, "dur_ms")?)?,
                        depth(record)?,
                        raw
                    ])
                    .map_err(sql_error)?;
                Ok(())
            })?;
            tracks.push(mapping.unbind());
        }
    }
    db.execute_batch("COMMIT; CREATE INDEX layout ON spans(pid,track,start,duration DESC,length(depth),depth,id); CREATE INDEX output ON spans(pid,id);").map_err(sql_error)?;
    if origin == f64::INFINITY {
        origin = 0.0;
    }
    if let Some(observer) = observer {
        observer.call1(("layout", py.None()))?;
    }
    db.execute_batch("BEGIN").map_err(sql_error)?;
    let mut count = 0;
    for (pid, (_, name)) in files.iter().enumerate() {
        let name = PyString::new(py, name);
        emit(&metadata(py, "process_name", pid, 0, name.as_any())?)?;
        let mapping = tracks[pid].bind(py);
        let mut offset = 0;
        for track in sorted_tracks(py, mapping)? {
            let track_id: i64 = mapping.get_item(&track)?.unwrap().extract()?;
            let mut pool = LanePool::default();
            let mut query = db.prepare("SELECT id,start,duration,depth FROM spans WHERE pid=? AND track=? ORDER BY start,duration DESC,length(depth),depth,id").map_err(sql_error)?;
            let rows = query
                .query_map(params![pid as i64, track_id], |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                })
                .map_err(sql_error)?;
            let mut update = db
                .prepare("UPDATE spans SET lane=? WHERE id=?")
                .map_err(sql_error)?;
            for row in rows {
                let (id, start, duration, depth) = row.map_err(sql_error)?;
                let lane = offset + pool.place_inner(relative(start, origin)?, duration, depth)?;
                update
                    .execute(params![lane as i64, id])
                    .map_err(sql_error)?;
            }
            let label = if mapping.len() > 1 {
                PyString::new(py, "thread ").call_method1("__add__", (track.str()?,))?
            } else {
                name.clone().into_any()
            };
            for lane in 0..pool.stacks.len() {
                emit(&metadata(py, "thread_name", pid, offset + lane, &label)?)?;
            }
            offset += pool.stacks.len();
        }
        let mut query = db
            .prepare("SELECT record,lane FROM spans WHERE pid=? ORDER BY id")
            .map_err(sql_error)?;
        let rows = query
            .query_map([pid as i64], |row| {
                Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(sql_error)?;
        let kw = PyDict::new(py);
        kw.set_item("parse_constant", py.import("builtins")?.getattr("str")?)?;
        kw.set_item("parse_float", wrap_pyfunction!(diagnostic_float, py)?)?;
        let loads = py.import("json")?.getattr("loads")?;
        for row in rows {
            let (payload, lane) = row.map_err(sql_error)?;
            let record = loads.call((PyBytes::new(py, &payload),), Some(&kw))?;
            let record = record.cast::<PyDict>()?;
            let args = PyDict::new(py);
            for key in ["args", "metrics"] {
                let value = get(record, key)?;
                if !value.is_none() {
                    args.update(value.cast::<PyDict>()?.as_mapping())?;
                }
            }
            args.set_item("step", get(record, "_step")?)?;
            let error = get(record, "error")?;
            if error.is_truthy()? {
                args.set_item("error", error)?;
            }
            if mapping.len() > 1 {
                args.set_item(
                    "track",
                    record
                        .get_item("track")?
                        .unwrap_or_else(|| 0_i64.into_pyobject(py).unwrap().into_any()),
                )?;
            }
            let event = PyDict::new(py);
            event.set_item(
                "name",
                record
                    .get_item("name")?
                    .unwrap_or_else(|| PyString::new(py, "span").into_any()),
            )?;
            event.set_item("cat", &name)?;
            event.set_item("ph", "X")?;
            event.set_item("pid", pid)?;
            event.set_item("tid", lane)?;
            event.set_item("ts", relative(number(&get(record, "start")?)?, origin)?)?;
            event.set_item("dur", number(&get(record, "dur_ms")?)? * 1000.0)?;
            event.set_item("args", args)?;
            if let Some(observer) = observer {
                observer.call1(("event", &event))?;
            }
            emit(&event)?;
            count += 1;
        }
    }
    db.execute_batch("COMMIT").map_err(sql_error)?;
    Ok(count)
}

#[pyfunction]
#[pyo3(signature=(files, step_range=None, time_range=None, temp_dir=None, observer=None))]
fn trace_build(
    py: Python<'_>,
    files: Vec<(Py<PyAny>, String)>,
    step_range: Option<&Bound<'_, PyAny>>,
    time_range: Option<&Bound<'_, PyAny>>,
    temp_dir: Option<&Bound<'_, PyAny>>,
    observer: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyDict>> {
    let files = files
        .into_iter()
        .map(|(p, n)| Ok((crate::artifacts::path_value(p.bind(py), false)?, n)))
        .collect::<PyResult<Vec<_>>>()?;
    let filters = Filters::new(py, step_range, time_range)?;
    let directory = temp_dir
        .filter(|v| !v.is_none())
        .map(|v| crate::artifacts::path_value(v, false))
        .transpose()?
        .unwrap_or(std::env::current_dir()?);
    let events = PyList::empty(py);
    export(py, files, &filters, &directory, observer, |event| {
        events.append(event)
    })?;
    let result = PyDict::new(py);
    result.set_item("traceEvents", events)?;
    result.set_item("displayTimeUnit", "ms")?;
    Ok(result.unbind())
}

#[pyfunction]
#[pyo3(signature=(files, output, step_range=None, time_range=None, temp_dir=None, observer=None))]
fn trace_write(
    py: Python<'_>,
    files: Vec<(Py<PyAny>, String)>,
    output: &Bound<'_, PyAny>,
    step_range: Option<&Bound<'_, PyAny>>,
    time_range: Option<&Bound<'_, PyAny>>,
    temp_dir: Option<&Bound<'_, PyAny>>,
    observer: Option<&Bound<'_, PyAny>>,
) -> PyResult<usize> {
    let files = files
        .into_iter()
        .map(|(p, n)| Ok((crate::artifacts::path_value(p.bind(py), false)?, n)))
        .collect::<PyResult<Vec<_>>>()?;
    let output = crate::artifacts::path_value(output, false)?;
    let filters = Filters::new(py, step_range, time_range)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let directory = temp_dir
        .filter(|v| !v.is_none())
        .map(|v| crate::artifacts::path_value(v, false))
        .transpose()?
        .unwrap_or(parent.to_owned());
    let mut target = tempfile::Builder::new()
        .prefix(".expr-trace-")
        .suffix(".tmp")
        .tempfile_in(parent)?;
    let encoder = py.import("json")?.getattr("dumps")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("ensure_ascii", true)?;
    kwargs.set_item("allow_nan", false)?;
    let count = {
        let mut writer = BufWriter::with_capacity(64 << 10, &mut target);
        writer.write_all(b"{\"traceEvents\":[")?;
        let mut first = true;
        let count = export(py, files, &filters, &directory, observer, |event| {
            if !first {
                writer.write_all(b",")?;
            }
            first = false;
            let encoded: String = encoder.call((event,), Some(&kwargs))?.extract()?;
            writer.write_all(encoded.as_bytes())?;
            Ok(())
        })?;
        writer.write_all(b"],\"displayTimeUnit\":\"ms\"}")?;
        writer.flush()?;
        count
    };
    target.persist(output).map_err(|e| PyErr::from(e.error))?;
    Ok(count)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<LanePool>()?;
    module.add_function(wrap_pyfunction!(trace_read_spans, module)?)?;
    module.add_function(wrap_pyfunction!(trace_assign_lanes, module)?)?;
    module.add_function(wrap_pyfunction!(trace_build, module)?)?;
    module.add_function(wrap_pyfunction!(trace_write, module)?)
}
