use super::{
    callback, duration, segment_paths, write_all_slices, File, Instant, IoSlice, NativeWriter,
    OpenOptions, PyBytes, Queued, VecDeque,
};
use pyo3::prelude::*;
use std::fs;
use std::io::{self, Write};

#[pymethods]
impl NativeWriter {
    pub(super) fn schedule(&self, _py: Python<'_>, start_worker: Py<PyAny>) -> PyResult<()> {
        self.check()?;
        let mut state = self.state.lock();
        let before = (state.deadline, state.requested);
        let start = state.schedule(false);
        let wake = start || (state.deadline, state.requested) != before;
        drop(state);
        if wake {
            self.condition.notify_all();
        }
        if start {
            callback(&start_worker, ())?;
        }
        Ok(())
    }
    pub(super) fn wait_work(&self, py: Python<'_>) -> PyResult<bool> {
        self.check()?;
        Ok(py.detach(|| {
            let mut s = self.state.lock();
            loop {
                if s.closing {
                    return false;
                }
                let now = Instant::now();
                let deadline = if s.requested { Some(now) } else { s.deadline };
                if let Some(deadline) = deadline {
                    let deadline = s.retry.map_or(deadline, |retry| deadline.max(retry));
                    if deadline <= now {
                        return true;
                    }
                    self.condition
                        .wait_for(&mut s, deadline.duration_since(now));
                } else {
                    self.condition.wait(&mut s);
                }
            }
        }))
    }
    #[pyo3(signature=(payload,append=None))]
    pub(super) fn write(
        &self,
        py: Python<'_>,
        payload: &[u8],
        append: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        self.check()?;
        py.detach(|| self.write_data(payload, append.as_ref()))
    }
    pub(super) fn append(&self, py: Python<'_>, payload: &[u8]) -> PyResult<()> {
        self.check()?;
        Ok(py.detach(|| self.append_data(payload))?)
    }
    pub(super) fn rollback(&self, py: Python<'_>, size: u64) -> PyResult<bool> {
        self.check()?;
        Ok(py.detach(|| self.rollback_data(size))?)
    }
    pub(super) fn flush(
        &self,
        py: Python<'_>,
        write: Option<Py<PyAny>>,
        rollback: Option<Py<PyAny>>,
        save: Option<Py<PyAny>>,
    ) -> PyResult<Vec<(String, String)>> {
        self.check()?;
        py.detach(|| {
            let _serial = self.write_lock.lock();
            let mut diagnostics = Vec::new();
            let (mut records, size) = {
                let mut s = self.state.lock();
                if s.closed {
                    return Ok(diagnostics);
                }
                s.requested = false;
                s.deadline = None;
                s.first = None;
                if s.queue.is_empty() {
                    return Ok(diagnostics);
                }
                let records = std::mem::take(&mut s.queue);
                s.inflight = records.len();
                s.inflight_bytes = s.queued_bytes;
                s.queued_bytes = 0;
                (records, s.size)
            };
            let legacy_boundaries = self.state.lock().max_file.is_some()
                && records.iter().any(|record| {
                    let line = record.line.as_slice();
                    !line.ends_with(b"\n")
                        || memchr::memchr(b'\n', &line[..line.len() - 1]).is_some()
                });
            let result = if write.is_some() || legacy_boundaries {
                let mut payload = Vec::with_capacity(records.iter().map(|r| r.line.len()).sum());
                for record in &records {
                    payload.extend_from_slice(&record.line);
                }
                if let Some(write) = write {
                    Python::attach(|py| write.call1(py, (PyBytes::new(py, &payload),))).map(|_| ())
                } else {
                    self.write_data(&payload, None)
                }
            } else {
                self.write_records(&records).map_err(PyErr::from)
            };
            if let Err(error) = result {
                let error_path = self.state.lock().path.clone();
                diagnostics.push((
                    "error".into(),
                    format!(
                        "Failed to flush metrics to {}: {}",
                        error_path.display(),
                        error
                    ),
                ));
                let restored_result = if let Some(rollback) = rollback {
                    Python::attach(|py| {
                        rollback
                            .call1(py, (size,))
                            .and_then(|v| v.extract::<bool>(py))
                    })
                } else {
                    self.rollback_data(size).map_err(PyErr::from)
                };
                let restored = match restored_result {
                    Ok(restored) => restored,
                    Err(error) => {
                        let error_path = self.state.lock().path.clone();
                        diagnostics.push((
                            "warning".into(),
                            format!(
                                "Could not roll back partial write on {}: {}",
                                error_path.display(),
                                error
                            ),
                        ));
                        false
                    }
                };
                let mut s = self.state.lock();
                if !restored {
                    s.duplicates = true;
                }
                records.append(&mut s.queue);
                s.queue = records;
                s.queued_bytes += s.inflight_bytes;
                s.inflight_bytes = 0;
                s.inflight = 0;
                s.first = Some(Instant::now());
                s.retry = Some(Instant::now() + duration(s.max_seconds.unwrap_or(1.0).max(0.1)));
                let async_write = s.async_write;
                if s.schedule(async_write) {
                    // The SDK starts its callback trampoline after this native
                    // operation has released its serialized write lock.
                    s.worker_started = false;
                }
                self.condition.notify_all();
                return Ok(diagnostics);
            }
            {
                let mut s = self.state.lock();
                let mut offset = size;
                for r in records {
                    s.track(Some(r.step), offset);
                    s.flushed = s.flushed.max(r.row);
                    offset += r.line.len() as u64;
                }
                s.size = offset;
                s.inflight = 0;
                s.inflight_bytes = 0;
                s.retry = None;
                self.condition.notify_all();
            }
            if let Some(save) = save {
                callback(&save, ())?;
            } else {
                diagnostics.extend(self.save_meta_inner(false));
            }
            Ok(diagnostics)
        })
    }
    #[pyo3(signature=(force=false))]
    pub(super) fn save_meta(&self, py: Python<'_>, force: bool) -> PyResult<Vec<(String, String)>> {
        self.check()?;
        Ok(py.detach(|| self.save_meta_inner(force)))
    }
    pub(super) fn begin_close(&self, py: Python<'_>) -> PyResult<bool> {
        self.check()?;
        Ok(py.detach(|| {
            let mut s = self.state.lock();
            if s.closing {
                while !s.closed {
                    self.condition.wait(&mut s);
                }
                return false;
            }
            s.closing = true;
            s.requested = false;
            s.deadline = None;
            self.condition.notify_all();
            true
        }))
    }
    pub(super) fn end_close(&self, py: Python<'_>) -> PyResult<Vec<(String, String)>> {
        self.check()?;
        Ok(py.detach(||{
            let _serial=self.write_lock.lock();let mut diagnostics=Vec::new();
            let mut s=self.state.lock();if !s.queue.is_empty(){diagnostics.push(("error".into(),format!("Closing {} with {} unwritten records after a failed final flush ({} bytes).",s.path.display(),s.queue.len(),s.queued_bytes)));}
            s.closed=true;self.condition.notify_all();diagnostics
        }))
    }
}
impl NativeWriter {
    pub(super) fn write_records(&self, records: &VecDeque<Queued>) -> io::Result<()> {
        let mut start = 0;
        while start < records.len() {
            let (end, size, path) = {
                let mut s = self.state.lock();
                let mut end = start;
                let mut size = 0;
                while end < records.len() {
                    let next = records[end].line.len() as u64;
                    if s.max_file
                        .is_some_and(|max| s.active_size + size + next > max)
                        && (s.active_size > 0 || size > 0)
                    {
                        break;
                    }
                    size += next;
                    end += 1;
                }
                if end == start {
                    s.active += 1;
                    s.active_size = 0;
                    continue;
                }
                (end, size, s.active_path())
            };
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new().create(true).append(true).open(path)?;
            let mut offset = start;
            while offset < end {
                let next = (offset + 128).min(end);
                if next == offset + 1 {
                    file.write_all(&records[offset].line)?;
                } else {
                    let mut slices: Vec<_> = (offset..next)
                        .map(|index| IoSlice::new(records[index].line.as_slice()))
                        .collect();
                    write_all_slices(&mut file, &mut slices)?;
                }
                offset = next;
            }
            self.state.lock().active_size += size;
            start = end;
        }
        Ok(())
    }

    pub(super) fn write_data(&self, payload: &[u8], append: Option<&Py<PyAny>>) -> PyResult<()> {
        let mut start = 0;
        while start < payload.len() {
            let end = {
                let mut s = self.state.lock();
                let mut end = start;
                while end < payload.len() {
                    let next = memchr::memchr(b'\n', &payload[end..])
                        .map_or(payload.len(), |n| end + n + 1);
                    if s.max_file
                        .is_some_and(|max| s.active_size + (next - start) as u64 > max)
                        && (s.active_size > 0 || end > start)
                    {
                        break;
                    }

                    end = next;
                }
                if end == start {
                    s.active += 1;
                    s.active_size = 0;
                    continue;
                }
                end
            };
            if let Some(append) = append {
                Python::attach(|py| append.call1(py, (PyBytes::new(py, &payload[start..end]),)))?;
            } else {
                self.append_data(&payload[start..end])?;
            }
            self.state.lock().active_size += (end - start) as u64;
            start = end;
        }
        Ok(())
    }
    pub(super) fn append_data(&self, payload: &[u8]) -> io::Result<()> {
        let path = self.state.lock().active_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(&payload)
    }
    pub(super) fn rollback_data(&self, size: u64) -> io::Result<bool> {
        let path = self.state.lock().path.clone();
        let mut remaining = size;
        for part in segment_paths(&path)? {
            if !part.exists() {
                continue;
            }
            let length = fs::metadata(&part)?.len();
            if remaining >= length {
                remaining -= length;
            } else if remaining > 0 || part == path {
                OpenOptions::new()
                    .write(true)
                    .open(&part)?
                    .set_len(remaining)?;
                remaining = 0;
            } else {
                fs::remove_file(part)?;
            }
        }
        let parts = segment_paths(&path)?;
        let active = parts.last().unwrap();
        let mut s = self.state.lock();
        s.active = if *active == path {
            0
        } else {
            active
                .file_stem()
                .unwrap()
                .to_string_lossy()
                .parse()
                .unwrap_or(0)
        };
        s.active_size = fs::metadata(active).map_or(0, |m| m.len());
        Ok(true)
    }
    pub(super) fn save_meta_inner(&self, force: bool) -> Vec<(String, String)> {
        let (path, text) = {
            let mut s = self.state.lock();
            let now = Instant::now();
            if !force
                && s.last_meta
                    .is_some_and(|last| now.duration_since(last).as_secs_f64() < 2.0)
            {
                return Vec::new();
            }
            s.last_meta = Some(now);
            (s.meta_path.clone(), s.metadata())
        };
        let tmp = path.with_extension("json.tmp");
        let result = (|| -> io::Result<()> {
            let mut f = File::create(&tmp)?;
            f.write_all(text.as_bytes())?;
            fs::rename(&tmp, &path)
        })();
        match result {
            Ok(()) => Vec::new(),
            Err(e) => vec![(
                "warning".into(),
                format!("Failed to write {}: {}", path.display(), e),
            )],
        }
    }
}
