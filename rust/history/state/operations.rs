use super::{
    float_json, in_range, Arc, BTreeMap, Cached, HashMap, HashSet, HistoryQuery, JsonObject,
    NativeHistoryState, NativeRecord, PyDict, SegmentedReader, Step,
};
use pyo3::prelude::*;

#[pymethods]
impl NativeHistoryState {
    pub(super) fn record_span_native(
        &self,
        py: Python<'_>,
        span: &crate::spans::NativeSpan,
        name: &Bound<'_, pyo3::types::PyString>,
        depth: usize,
        track: &Bound<'_, pyo3::types::PyAny>,
        error: Option<&Bound<'_, pyo3::types::PyString>>,
        span_metrics: bool,
        span_count: bool,
        writer: Option<&super::super::writer::NativeWriter>,
        row: i64,
        start_worker: Py<PyAny>,
        wait_observer: Py<PyAny>,
        now: f64,
    ) -> PyResult<
        Option<(
            String,
            bool,
            bool,
            Vec<(String, String)>,
            Vec<(String, String)>,
        )>,
    > {
        let Ok(name) = name.to_str() else {
            return Ok(None);
        };
        if !track.is_exact_instance_of::<pyo3::types::PyInt>() {
            return Ok(None);
        }
        let Ok(track) = track.extract::<i64>() else {
            return Ok(None);
        };
        let error = match error {
            Some(error) => match error.to_str() {
                Ok(error) => Some(error),
                Err(_) => return Ok(None),
            },
            None => None,
        };
        let (started_at, duration) = span.timing();
        let (step, warnings) = if span_metrics {
            let mut metrics = JsonObject::default();
            metrics.set(format!("time_ms/{name}"), float_json(duration));
            if span_count {
                metrics.set(format!("count/{name}"), "1".into());
            }
            let (step, _, warnings) = self.ingest_object(
                py,
                metrics,
                None,
                false,
                true,
                HashMap::new(),
                duration,
                now,
            )?;
            (step.unwrap(), warnings)
        } else {
            (self.state.lock().current().0, Vec::new())
        };
        let Some(writer) = writer else {
            return Ok(Some((step, false, false, Vec::new(), warnings)));
        };
        let error = error
            .map(|error| format!(", \"error\": {}", serde_json::to_string(error).unwrap()))
            .unwrap_or_default();
        let line = format!(
            "{{\"_step\": {}, \"name\": {}, \"depth\": {}, \"track\": {}, \"start\": {}, \"dur_ms\": {}{}}}\n",
            step, serde_json::to_string(name).unwrap(), depth, track,
            float_json(started_at), float_json(duration), error,
        );
        let (accepted, due, diagnostics) = writer.enqueue_shared(
            py,
            step.clone(),
            row,
            Arc::new(line.into_bytes()),
            start_worker,
            wait_observer,
        )?;
        Ok(Some((
            step,
            accepted,
            due && !writer.asynchronous(),
            diagnostics,
            warnings,
        )))
    }
    pub(super) fn prepare_local(
        &self,
        py: Python<'_>,
        values: &Bound<'_, PyDict>,
        step: Option<String>,
        commit: bool,
        writer: &super::super::writer::NativeWriter,
        notifications: &super::super::notifications::NativeCommitQueue,
        start_worker: Py<PyAny>,
        wait_observer: Py<PyAny>,
        now: f64,
    ) -> PyResult<
        Option<(
            Option<String>,
            Vec<NativeRecord>,
            bool,
            bool,
            Vec<(String, String)>,
            Vec<(String, String)>,
        )>,
    > {
        if !notifications.idle() {
            return Ok(None);
        }
        let Some((mut resolved, records, warnings)) =
            self.ingest_values(py, values, step, commit, false, HashMap::new(), 1.0, now)?
        else {
            return Ok(None);
        };
        let mut accepted = Vec::with_capacity(records.len());
        let mut diagnostics = Vec::new();
        let mut flush = false;
        for record in records {
            let (admitted, due, messages) = self.enqueue_record(
                py,
                writer,
                &record,
                start_worker.clone_ref(py),
                wait_observer.clone_ref(py),
                now,
            )?;
            diagnostics.extend(messages);
            if admitted {
                flush |= due;
                accepted.push(record);
            } else if resolved.as_deref() == Some(record.step.0.as_str()) {
                resolved = None;
            }
        }
        let budget_met = self.evict_writer(writer);
        Ok(Some((
            resolved,
            accepted,
            flush,
            budget_met,
            diagnostics,
            warnings,
        )))
    }
    pub(super) fn publish_local(
        &self,
        py: Python<'_>,
        records: Vec<PyRef<'_, NativeRecord>>,
        series: &super::super::series::NativeMetricSeries,
        summary: Option<&crate::summary::NativeSummary>,
    ) -> Option<Vec<(String, String)>> {
        if summary.is_some_and(|summary| !summary.can_inline(py)) {
            return None;
        }
        for record in &records {
            if record.validated() {
                if !record.inline_numeric {
                    return None;
                }
                continue;
            }
            for (key, value) in record.values(py).iter() {
                let Ok(key) = key.cast::<pyo3::types::PyString>() else {
                    return None;
                };
                let Ok(name) = key.to_str() else {
                    return None;
                };
                if value.is_instance_of::<pyo3::types::PyInt>()
                    && !name.starts_with('_')
                    && value.extract::<f64>().is_err()
                {
                    return None;
                }
            }
        }
        let mut diagnostics = Vec::new();
        for record in records {
            let values = record.values(py);
            let result = if record.validated() {
                series.add_validated_record(py, record.step.0.clone(), record.timestamp, values)
            } else {
                series.add_record(py, record.step.0.clone(), record.timestamp, values)
            };
            match result {
                Ok(warnings) => {
                    for key in warnings {
                        diagnostics.push(("warning".into(), format!(
                            "Metric {key} cannot be represented as a float; skipping its numeric series."
                        )));
                    }
                }
                Err(error) => diagnostics.push((
                    "warning".into(),
                    format!("Could not update metric series: {error}"),
                )),
            }
            if let Some(summary) = summary {
                if let Err(error) = summary.observe(py, values.as_any()) {
                    diagnostics.push((
                        "warning".into(),
                        format!("Failed to update summary: {error}"),
                    ));
                }
            }
        }
        Some(diagnostics)
    }
    pub(super) fn enqueue_record(
        &self,
        py: Python<'_>,
        writer: &super::super::writer::NativeWriter,
        record: &NativeRecord,
        start_worker: Py<PyAny>,
        wait_observer: Py<PyAny>,
        now: f64,
    ) -> PyResult<(bool, bool, Vec<(String, String)>)> {
        let row = self.state.lock().row;
        let result = writer.enqueue_shared(
            py,
            record.step.0.clone(),
            row,
            record.line.clone(),
            start_worker,
            wait_observer,
        )?;
        if result.0 {
            let mut s = self.state.lock();
            if s.emitted
                .as_ref()
                .is_some_and(|previous| record.step <= *previous)
            {
                s.merge = true;
            }
            s.emitted = Some(record.step.clone());
            s.row += 1;
            s.bytes += record.line.len();
            s.cache.push_back(Cached {
                row,
                step: record.step.clone(),
                line: record.line.clone(),
            });
            s.last_commit = Some(now);
        }
        Ok(result)
    }
    pub(super) fn evict_writer(&self, writer: &super::super::writer::NativeWriter) -> bool {
        self.evict(writer.flushed())
    }
    pub(super) fn close_open(&self) -> Option<String> {
        self.state.lock().close().map(|r| r.encode())
    }
    pub(super) fn open_record(&self) -> Option<String> {
        self.state.lock().opened().map(|r| r.encode())
    }
    pub(super) fn admitted(&self, step: String, line: String, now: f64) {
        let mut s = self.state.lock();
        let step = Step(step);
        if s.emitted.as_ref().is_some_and(|previous| step <= *previous) {
            s.merge = true;
        }
        s.emitted = Some(step.clone());
        let row = s.row;
        s.row += 1;
        s.bytes += line.len();
        s.cache.push_back(Cached {
            row,
            step,
            line: Arc::new(line.into_bytes()),
        });
        s.last_commit = Some(now);
    }
    pub(super) fn evict(&self, flushed: i64) -> bool {
        let mut s = self.state.lock();
        while s.over() && s.cache.front().is_some_and(|r| r.row <= flushed) {
            let row = s.cache.pop_front().unwrap();
            s.bytes -= row.line.len();
            s.disk_prefix = true;
            s.evicted += 1;
        }
        !s.over()
    }
    pub(super) fn get(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        use pyo3::IntoPyObjectExt;
        let s = self.state.lock();
        match name {
            "current_step" => s.current().0.into_py_any(py),
            "_next_step" => s.next.0.clone().into_py_any(py),
            "_open_step" => s.open.as_ref().map(|s| s.0.clone()).into_py_any(py),
            "_open_row" => s.data.encode().into_py_any(py),
            "_open_weights" => s.weights.clone().into_py_any(py),
            "_open_time" => s.time.into_py_any(py),
            "_next_row" => s.row.into_py_any(py),
            "_last_committed_step" => s.last.as_ref().map(|s| s.0.clone()).into_py_any(py),
            "_last_emitted_step" => s.emitted.as_ref().map(|s| s.0.clone()).into_py_any(py),
            "_cache_bytes" => s.bytes.into_py_any(py),
            "_has_disk_prefix" => s.disk_prefix.into_py_any(py),
            "_needs_merge" => s.merge.into_py_any(py),
            "_evicted_rows" => s.evicted.into_py_any(py),
            "_queries" => s.queries.into_py_any(py),
            "_disk_queries" => s.disk_queries.into_py_any(py),
            "_last_commit_time" => s.last_commit.into_py_any(py),
            "_over_budget" => s.over().into_py_any(py),
            "_cache" => s
                .cache
                .iter()
                .map(|r| {
                    (
                        r.row,
                        r.step.0.clone(),
                        String::from_utf8_lossy(&r.line).into_owned(),
                    )
                })
                .collect::<Vec<_>>()
                .into_py_any(py),
            "cached_rows" => s.cache.len().into_py_any(py),
            _ => Err(pyo3::exceptions::PyAttributeError::new_err(name.to_owned())),
        }
    }
    pub(super) fn view_tail(
        &self,
        count: Option<usize>,
        merge: bool,
    ) -> (Vec<(i64, String, String)>, bool, i64) {
        let s = self.state.lock();
        let mut rows = Vec::new();
        let mut seen = HashSet::new();
        let mut complete = false;
        for row in s.cache.iter().rev() {
            if !merge && count.is_some_and(|n| seen.len() == n) && !seen.contains(&row.step) {
                complete = true;
                break;
            }
            seen.insert(row.step.clone());
            rows.push((
                row.row,
                row.step.0.clone(),
                String::from_utf8_lossy(&row.line).into_owned(),
            ));
        }
        rows.reverse();
        (
            rows,
            complete || !s.disk_prefix,
            s.cache.front().map_or(s.row, |r| r.row),
        )
    }
    #[pyo3(signature=(writer,count=None,start=None,end=None,include_open=true,ranged=false))]
    pub(super) fn collect(
        &self,
        py: Python<'_>,
        writer: &super::super::writer::NativeWriter,
        count: Option<usize>,
        start: Option<String>,
        end: Option<String>,
        include_open: bool,
        ranged: bool,
    ) -> PyResult<(Vec<String>, Option<String>)> {
        let start = start.map(Step);
        let end = end.map(Step);
        if count == Some(0) {
            self.state.lock().queries += 1;
            return Ok((Vec::new(), None));
        }
        let (path, durable_size, dropped, index, disk_merge, rows, opened, merge, boundary, prefix) =
            writer.snapshot(|path, durable_size, dropped, index, disk_merge| {
                let mut s = self.state.lock();
                s.queries += 1;
                let opened = if include_open { s.opened() } else { None };
                let merge = s.merge
                    || dropped
                    || opened.as_ref().is_some_and(|r| {
                        s.emitted
                            .as_ref()
                            .is_some_and(|last| r.step().unwrap() <= *last)
                    });
                let mut complete = !s.disk_prefix;
                let mut rows = Vec::new();
                let mut seen = HashSet::new();
                if !ranged {
                    for row in s.cache.iter().rev() {
                        if !merge
                            && count.is_some_and(|n| seen.len() == n)
                            && !seen.contains(&row.step)
                        {
                            complete = true;
                            break;
                        }
                        seen.insert(row.step.clone());
                        rows.push(row.clone());
                    }
                    rows.reverse();
                } else {
                    rows = s
                        .cache
                        .iter()
                        .filter(|r| in_range(&r.step, &start, &end))
                        .cloned()
                        .collect();
                    if !merge
                        && start
                            .as_ref()
                            .is_some_and(|start| s.cache.front().is_some_and(|r| *start >= r.step))
                    {
                        complete = true;
                    }
                }
                if !complete {
                    s.disk_queries += 1;
                }
                (
                    path,
                    durable_size,
                    dropped,
                    index,
                    disk_merge,
                    rows,
                    opened,
                    merge,
                    s.cache.front().map_or(s.row, |r| r.row),
                    !complete,
                )
            });
        // Complete small cache views require neither I/O nor blocking waits.
        let cheap = !prefix
            && !merge
            && rows.len() <= 512
            && opened.as_ref().is_none_or(|row| {
                row.0
                    .iter()
                    .map(|(key, value)| key.len() + value.len())
                    .sum::<usize>()
                    <= 64 * 1024
            });
        let collect = || {
            let mut records = Vec::new();
            let mut warning = None;
            if !prefix && !merge {
                let mut encoded: Vec<String> = rows
                    .into_iter()
                    .map(|row| String::from_utf8_lossy(&row.line).into_owned())
                    .collect();
                if let Some(opened) =
                    opened.filter(|record| in_range(&record.step().unwrap(), &start, &end))
                {
                    encoded.push(opened.encode());
                }
                if let Some(count) = count {
                    if encoded.len() > count {
                        encoded.drain(..encoded.len() - count);
                    }
                }
                return (encoded, None);
            }
            let newer: Vec<_> = rows
                .iter()
                .filter_map(|r| {
                    JsonObject::parse(std::str::from_utf8(&r.line).expect("cached JSON is UTF-8"))
                })
                .collect();
            let opened = opened.filter(|r| in_range(&r.step().unwrap(), &start, &end));
            if prefix {
                let old = (|| -> std::io::Result<Vec<JsonObject>> {
                    let reader = SegmentedReader::open(&path)?
                        .snapshot(Some(durable_size))
                        .with_hints(index, disk_merge);
                    let stop = if dropped {
                        durable_size
                    } else {
                        reader.offset_line(boundary.max(0) as usize)?
                    };
                    let reader = reader.snapshot(Some(stop));
                    if ranged {
                        return reader.query(&HistoryQuery {
                            limit: None,
                            start: start.clone(),
                            end: end.clone(),
                            merge: Some(merge),
                        });
                    }
                    if let Some(count) = count {
                        if !merge {
                            return reader.query(&HistoryQuery {
                                limit: Some(
                                    count.saturating_sub(
                                        newer.len() + usize::from(opened.is_some()),
                                    ),
                                ),
                                merge: Some(false),
                                ..Default::default()
                            });
                        }
                        let mut selected = HashSet::new();
                        if let Some(record) = &opened {
                            selected.insert(record.step().unwrap());
                        }
                        for record in newer.iter().rev() {
                            if selected.len() >= count {
                                break;
                            }
                            selected.insert(record.step().unwrap());
                        }
                        if selected.len() < count {
                            reader.reverse(|row| {
                                if let Some(record) = super::super::reader::parse_row(&row.bytes) {
                                    selected.insert(record.step().unwrap());
                                }
                                Ok(selected.len() < count)
                            })?;
                        }
                        let mut old = BTreeMap::<Step, JsonObject>::new();
                        reader.visit(0, true, |row| {
                            if let Some(record) = super::super::reader::parse_row(&row.bytes) {
                                let step = record.step().unwrap();
                                if selected.contains(&step) {
                                    old.entry(step).or_default().merge(&record);
                                }
                            }
                            Ok(())
                        })?;
                        return Ok(old.into_values().collect());
                    }
                    reader.query(&HistoryQuery {
                        merge: Some(false),
                        ..Default::default()
                    })
                })();
                match old {
                    Ok(old) => records = old,
                    Err(error) => {
                        warning = Some(format!(
                            "Could not read older history from {}: {}; returning cached rows only.",
                            path.display(),
                            error
                        ))
                    }
                }
            }
            records.extend(newer);
            records.extend(opened);
            if !ranged {
                if let Some(count) = count {
                    let mut selected = HashSet::new();
                    for record in records.iter().rev() {
                        selected.insert(record.step().unwrap());
                        if selected.len() == count {
                            break;
                        }
                    }
                    records.retain(|r| selected.contains(&r.step().unwrap()));
                }
            }
            if merge {
                let mut merged = BTreeMap::<Step, JsonObject>::new();
                for record in records {
                    merged
                        .entry(record.step().unwrap())
                        .or_default()
                        .merge(&record);
                }
                records = merged.into_values().collect();
            }
            if ranged {
                if let Some(count) = count {
                    if records.len() > count {
                        records.drain(..records.len() - count);
                    }
                }
            }
            (records.iter().map(JsonObject::encode).collect(), warning)
        };
        Ok(if cheap { collect() } else { py.detach(collect) })
    }
}
