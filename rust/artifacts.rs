use fs2::FileExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

thread_local! {
    static PUBLISHING: RefCell<(u32, BTreeSet<PathBuf>)> = const { RefCell::new((0, BTreeSet::new())) };
}
struct PublishGuard(PathBuf);
impl PublishGuard {
    fn enter(root: PathBuf) -> PyResult<Self> {
        PUBLISHING.with(|active| {
            let mut active = active.borrow_mut();
            if active.0 != std::process::id() {
                active.0 = std::process::id();
                active.1.clear();
            }
            if !active.1.insert(root.clone()) {
                return Err(PyRuntimeError::new_err(
                    "Cannot reenter artifact publication",
                ));
            }
            Ok(Self(root))
        })
    }
}
impl Drop for PublishGuard {
    fn drop(&mut self) {
        PUBLISHING.with(|active| {
            active.borrow_mut().1.remove(&self.0);
        });
    }
}

fn validate(path: &Path, label: &str) -> PyResult<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(PyValueError::new_err(format!(
            "Invalid artifact {label}: paths must be relative and cannot contain '.' or '..'"
        )));
    }
    Ok(())
}

fn text_bytes(value: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    Ok(value
        .call_method1("encode", ("utf-8", "surrogatepass"))?
        .cast::<PyBytes>()?
        .as_bytes()
        .to_vec())
}
pub(crate) fn path_value(value: &Bound<'_, PyAny>, logical: bool) -> PyResult<PathBuf> {
    let py = value.py();
    let path = py.import("os")?.getattr("fspath")?.call1((value,))?;
    let bytes = if let Ok(bytes) = path.cast::<PyBytes>() {
        bytes.as_bytes().to_vec()
    } else if logical {
        text_bytes(&path)?
    } else {
        py.import("os")?
            .getattr("fsencode")?
            .call1((path,))?
            .cast::<PyBytes>()?
            .as_bytes()
            .to_vec()
    };
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStringExt;
        Ok(std::ffi::OsString::from_vec(bytes).into())
    }
    #[cfg(not(unix))]
    {
        String::from_utf8(bytes)
            .map(PathBuf::from)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn artifact_validate_path(path: &Bound<'_, PyAny>) -> PyResult<()> {
    validate(&path_value(path, true)?, "path")
}

#[pyfunction]
fn artifact_file_digest(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<String> {
    let path = path_value(path, false)?;
    py.detach(|| hash_file(&path)).map_err(Into::into)
}

#[pyfunction]
fn artifact_append_lineage(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    line: &Bound<'_, PyBytes>,
) -> PyResult<()> {
    let path = path_value(path, false)?;
    let mut payload = line.as_bytes().to_vec();
    if payload.last() != Some(&b'\n') {
        payload.push(b'\n');
    }
    py.detach(|| -> std::io::Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .append(true)
            .open(path)?;
        file.lock_exclusive()?;
        let original_size = file.metadata()?.len();
        let result = file.write_all(&payload);
        if result.is_err() {
            let _ = file.set_len(original_size);
        }
        let _ = FileExt::unlock(&file);
        result
    })?;
    Ok(())
}

#[pyfunction]
fn artifact_file_entry(py: Python<'_>, path: &Bound<'_, PyAny>) -> PyResult<(u64, String)> {
    let path = path_value(path, false)?;
    if !path.is_file() {
        return Err(pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Artifact file does not exist: {}",
            path.display()
        )));
    }
    py.detach(|| -> std::io::Result<_> { Ok((fs::metadata(&path)?.len(), hash_file(&path)?)) })
        .map_err(Into::into)
}

#[pyfunction]
#[pyo3(signature=(root, prefix=None))]
fn artifact_directory_entries(
    py: Python<'_>,
    root: &Bound<'_, PyAny>,
    prefix: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyList>> {
    let root = path_value(root, false)?;
    if !root.is_dir() {
        return Err(pyo3::exceptions::PyNotADirectoryError::new_err(format!(
            "Artifact directory does not exist: {}",
            root.display()
        )));
    }
    fn walk(
        root: &Path,
        directory: &Path,
        rows: &mut Vec<(PathBuf, PathBuf, u64, String)>,
    ) -> std::io::Result<()> {
        for entry in fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                walk(root, &path, rows)?;
            } else if path.is_file() {
                rows.push((
                    path.strip_prefix(root).unwrap().to_owned(),
                    path.clone(),
                    fs::metadata(&path)?.len(),
                    hash_file(&path)?,
                ));
            }
        }
        Ok(())
    }
    let rows = py.detach(|| -> std::io::Result<_> {
        let mut rows = vec![];
        walk(&root, &root, &mut rows)?;
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(rows)
    })?;
    let out = PyList::empty(py);
    let prefix = match prefix
        .filter(|v| !v.is_none())
        .filter(|v| v.is_truthy().unwrap_or(false))
    {
        Some(prefix) => prefix
            .call_method1("rstrip", ("/",))?
            .call_method1("__add__", ("/",))?,
        None => PyString::new(py, "").into_any(),
    };
    for (logical, source, size, digest) in rows {
        #[cfg(unix)]
        let (logical, source) = {
            use std::os::unix::ffi::OsStrExt;
            let decode = py.import("os")?.getattr("fsdecode")?;
            (
                decode.call1((PyBytes::new(py, logical.as_os_str().as_bytes()),))?,
                decode.call1((PyBytes::new(py, source.as_os_str().as_bytes()),))?,
            )
        };
        #[cfg(not(unix))]
        let (logical, source) = (
            PyString::new(py, &logical.to_string_lossy().replace('\\', "/")).into_any(),
            PyString::new(py, &source.to_string_lossy()).into_any(),
        );
        let logical = prefix.call_method1("__add__", (logical,))?;
        validate(&path_value(&logical, true)?, "entry path")?;
        out.append((logical, size, digest, source))?;
    }
    Ok(out.unbind())
}

fn hash_file(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let size = file.metadata()?.len().clamp(1, 1 << 20) as usize;
    let mut buffer = vec![0; size];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

fn copy_file(source: &Path, target: &Path) -> std::io::Result<()> {
    if let (Ok(source_metadata), Ok(target_metadata)) = (fs::metadata(source), fs::metadata(target))
    {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            if source_metadata.dev() == target_metadata.dev()
                && source_metadata.ino() == target_metadata.ino()
            {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "Artifact source and destination are the same file",
                ));
            }
        }
        #[cfg(not(unix))]
        if fs::canonicalize(source)? == fs::canonicalize(target)? {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Artifact source and destination are the same file",
            ));
        }
    }
    fs::copy(source, target)?;
    copy_metadata(source, target)
}

fn copy_metadata(source: &Path, target: &Path) -> std::io::Result<()> {
    let metadata = fs::metadata(source)?;
    fs::set_permissions(target, metadata.permissions())?;
    let mut times = fs::FileTimes::new();
    if let Ok(modified) = metadata.modified() {
        times = times.set_modified(modified);
    }
    if let Ok(accessed) = metadata.accessed() {
        times = times.set_accessed(accessed);
    }
    File::open(target)?.set_times(times)?;
    Ok(())
}

fn dumps(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<String> {
    let kw = PyDict::new(py);
    kw.set_item("ensure_ascii", true)?;
    py.import("json")?
        .getattr("dumps")?
        .call((data,), Some(&kw))?
        .extract()
}

fn field<'py>(data: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    data.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Artifact index record missing {key:?}")))
}

fn aliases(data: &Bound<'_, PyDict>) -> PyResult<BTreeSet<Vec<u8>>> {
    let mut result = BTreeSet::new();
    if let Some(values) = data.get_item("aliases")? {
        for value in values.try_iter()? {
            let value = text_bytes(&value?)?;
            if value != b"latest" {
                result.insert(value);
            }
        }
    }
    Ok(result)
}
fn set_aliases(data: &Bound<'_, PyDict>, aliases: BTreeSet<Vec<u8>>) -> PyResult<()> {
    let py = data.py();
    let values = PyList::empty(py);
    for value in aliases {
        values
            .append(PyBytes::new(py, &value).call_method1("decode", ("utf-8", "surrogatepass"))?)?;
    }
    data.set_item("aliases", values)
}

fn digest(py: Python<'_>, data: &Bound<'_, PyDict>) -> PyResult<String> {
    let mut hash = Sha256::new();
    let kind = field(data, "type")?;
    let text = kind.call_method1("__add__", ("\0",))?;
    let bytes = text.call_method1("encode", ("utf-8", "surrogatepass"))?;
    hash.update(bytes.cast::<PyBytes>()?.as_bytes());
    let mut entries = field(data, "entries")?
        .cast::<PyList>()?
        .iter()
        .map(|entry| {
            let entry = entry.cast::<PyDict>()?.clone();
            Ok((text_bytes(&field(&entry, "path")?)?, entry))
        })
        .collect::<PyResult<Vec<_>>>()?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    for (_, entry) in entries {
        let parts = PyList::empty(py);
        parts.append(field(&entry, "path")?)?;
        parts.append(field(&entry, "digest")?)?;
        let reference = field(&entry, "ref")?;
        parts.append(if reference.is_none() {
            PyString::new(py, "").into_any()
        } else {
            reference
        })?;
        parts.append("")?;
        let text = PyString::new(py, "\0").call_method1("join", (parts,))?;
        let bytes = text.call_method1("encode", ("utf-8", "surrogatepass"))?;
        hash.update(bytes.cast::<PyBytes>()?.as_bytes());
    }
    Ok(format!("sha256:{:x}", hash.finalize()))
}

#[pyclass]
pub struct NativeArtifactStore {
    root: PathBuf,
    owner: u32,
}

#[pymethods]
impl NativeArtifactStore {
    #[new]
    fn new(root: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            root: path_value(root, false)?,
            owner: std::process::id(),
        })
    }

    fn entries(&self, py: Python<'_>) -> PyResult<(Py<PyList>, Vec<String>)> {
        self.check()?;
        self.read(py)
    }

    #[pyo3(signature=(data, mode, run, step, on_locked=None, link_errno=None))]
    fn log(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyDict>,
        mode: &str,
        run: &Bound<'_, PyAny>,
        step: &Bound<'_, PyAny>,
        on_locked: Option<&Bound<'_, PyAny>>,
        link_errno: Option<i32>,
    ) -> PyResult<(Py<PyDict>, Vec<String>)> {
        self.check()?;
        if !matches!(mode, "copy" | "link" | "reference") {
            return Err(PyValueError::new_err(format!(
                "Unknown artifact mode {mode:?}; use link, copy or reference"
            )));
        }
        let name = path_value(&field(data, "name")?, mode == "reference")?;
        validate(&name, "name")?;
        let inputs = field(data, "entries")?;
        let inputs = inputs.cast::<PyList>()?;
        let mut logical_paths = BTreeSet::new();
        for entry in inputs.iter() {
            let entry = entry.cast::<PyDict>()?;
            let path = path_value(&field(entry, "path")?, true)?;
            validate(&path, "entry path")?;
            if !logical_paths.insert(path) {
                return Err(PyValueError::new_err("Duplicate artifact entry path"));
            }
        }
        let hash = digest(py, data)?;
        let mut requested_aliases = aliases(data)?;
        data.set_item("digest", &hash)?;
        set_aliases(data, requested_aliases.clone())?;
        fs::create_dir_all(&self.root)?;
        let _guard = PublishGuard::enter(fs::canonicalize(&self.root)?)?;
        let lock = py.detach(|| -> std::io::Result<File> {
            fs::create_dir_all(&self.root)?;
            let lock = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(self.root.join(".lock"))?;
            lock.lock_exclusive()?;
            Ok(lock)
        })?;
        let result = (|| -> PyResult<(Py<PyDict>, Vec<String>)> {
            if let Some(callback) = on_locked {
                callback.call0()?;
            }
            let (records, diagnostics) = self.read(py)?;
            let mut existing = None;
            let mut next = 0_u64;
            for row in records.bind(py).iter() {
                let row = row.cast::<PyDict>()?;
                if !field(row, "name")?.eq(field(data, "name")?)? {
                    continue;
                }
                let version: u64 = field(row, "version")?.extract()?;
                next = next.max(
                    version
                        .checked_add(1)
                        .ok_or_else(|| PyValueError::new_err("Artifact version overflow"))?,
                );
                if field(row, "digest")?.eq(&hash)? {
                    existing = Some(row.copy()?);
                }
            }
            if let Some(existing) = existing {
                requested_aliases.extend(aliases(&existing)?);
                set_aliases(&existing, requested_aliases)?;
                existing.set_item("reused", true)?;
                self.append(py, &existing)?;
                return Ok((existing.unbind(), diagnostics));
            }
            let record = data.copy()?;
            record.set_item("version", next)?;
            record.set_item("digest", hash)?;
            set_aliases(&record, requested_aliases)?;
            record.set_item(
                "created_at",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
            )?;
            record.set_item("run", run)?;
            record.set_item("step", step)?;
            record.set_item("reused", false)?;
            let destination = self.root.join(&name).join(format!("v{next}"));
            let mut publication = None;
            if mode != "reference" {
                let parent = destination.parent().unwrap();
                fs::create_dir_all(parent)?;
                let staged = tempfile::Builder::new()
                    .prefix(".staging-")
                    .tempdir_in(parent)?;
                for entry in inputs.iter() {
                    let entry = entry.cast::<PyDict>()?;
                    if !field(entry, "ref")?.is_none() {
                        continue;
                    }
                    let Some(source) = entry.get_item("source")? else {
                        continue;
                    };
                    if source.is_none() {
                        continue;
                    }
                    let source = path_value(&source, false)?;
                    let logical = path_value(&field(entry, "path")?, false)?;
                    let target = staged.path().join(logical);
                    // Check bytes against the captured digest before publication:
                    // a checkpoint changed between add_file() and log() is not
                    // allowed to publish under the old content identity.
                    let expected: String = field(entry, "digest")?.extract()?;
                    py.detach(|| -> std::io::Result<()> {
                        fs::create_dir_all(target.parent().unwrap())?;
                        let copied = if mode == "link" {
                            let linked = match link_errno {
                                Some(code) => Err(std::io::Error::from_raw_os_error(code)),
                                None => fs::hard_link(&source, &target),
                            };
                            if linked.is_err() {
                                copy_file(&source, &target)?;
                                true
                            } else {
                                false
                            }
                        } else {
                            copy_file(&source, &target)?;
                            true
                        };
                        if hash_file(&target)? != expected {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "Artifact source changed after add_file",
                            ));
                        }
                        // Digest verification reads the destination and can
                        // update its atime; restore copy2 metadata afterwards.
                        if copied {
                            copy_metadata(&source, &target)?;
                        }
                        Ok(())
                    })?;
                }
                if destination.exists() {
                    return Err(PyRuntimeError::new_err(format!(
                        "Artifact version directory already exists without an index record: {}",
                        destination.display()
                    )));
                }
                fs::rename(staged.path(), &destination)?;
                publication = Some(destination);
            }
            let entries = PyList::empty(py);
            for entry in inputs.iter() {
                let clean = entry.cast::<PyDict>()?.copy()?;
                if clean.contains("source")? {
                    clean.del_item("source")?;
                }
                entries.append(clean)?;
            }
            record.set_item("entries", entries)?;
            if let Err(error) = self.append(py, &record) {
                if let Some(path) = publication {
                    let _ = fs::remove_dir_all(path);
                }
                return Err(error);
            }
            Ok((record.unbind(), diagnostics))
        })();
        let _ = FileExt::unlock(&lock);
        result
    }

    fn resolve(
        &self,
        py: Python<'_>,
        spec: &Bound<'_, PyAny>,
    ) -> PyResult<(Option<Py<PyDict>>, Vec<String>)> {
        self.check()?;
        let parts = spec.call_method1("partition", (":",))?;
        let name = parts.get_item(0)?;
        let selector = text_bytes(&parts.get_item(2)?)?;
        let selector = if selector.is_empty() {
            b"latest".to_vec()
        } else {
            selector
        };
        let (records, diagnostics) = self.read(py)?;
        let mut found = None;
        for row in records.bind(py).iter() {
            let row = row.cast::<PyDict>()?;
            if !field(row, "name")?.eq(&name)? {
                continue;
            }
            let selected = if selector == b"latest" {
                true
            } else if let Some(version) = selector
                .strip_prefix(b"v")
                .and_then(|v| std::str::from_utf8(v).ok())
                .and_then(|v| v.parse::<u64>().ok())
            {
                field(row, "version")?.extract::<u64>()? == version
            } else {
                aliases(row)?.contains(&selector)
            };
            if selected {
                found = Some(row.copy()?.unbind());
            }
        }
        Ok((found, diagnostics))
    }
}

impl NativeArtifactStore {
    fn check(&self) -> PyResult<()> {
        if self.owner != std::process::id() {
            return Err(PyRuntimeError::new_err(
                "Artifact store belongs to another process; create a new store after fork.",
            ));
        }
        Ok(())
    }

    fn read(&self, py: Python<'_>) -> PyResult<(Py<PyList>, Vec<String>)> {
        let records = PyList::empty(py);
        let mut known: HashMap<(Vec<u8>, u64), Py<PyDict>> = HashMap::new();
        let path = self.root.join("index.jsonl");
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok((records.unbind(), vec![]))
            }
            Err(e) => return Err(e.into()),
        };
        let loads = py.import("json")?.getattr("loads")?;
        let mut diagnostics = Vec::new();
        for (index, line) in BufReader::new(file).split(b'\n').enumerate() {
            let line = line?;
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            let decoded = (|| -> PyResult<()> {
                let row = loads.call1((PyBytes::new(py, &line),))?;
                let row = row.cast::<PyDict>()?;
                let name = text_bytes(&field(row, "name")?)?;
                validate(&path_value(&field(row, "name")?, true)?, "name")?;
                let version: u64 = field(row, "version")?.extract()?;
                let key = (name, version);
                let entries = field(row, "entries")?;
                for entry in entries.cast::<PyList>()?.iter() {
                    let entry = entry.cast::<PyDict>()?;
                    validate(&path_value(&field(entry, "path")?, true)?, "entry path")?;
                }
                if let Some(metadata) = row.get_item("metadata")? {
                    if !metadata.is_none() {
                        metadata.cast::<PyDict>()?;
                    }
                }
                if !row.contains("digest")? {
                    row.set_item("digest", py.None())?;
                }
                let reused = row
                    .get_item("reused")?
                    .is_some_and(|v| v.is_truthy().unwrap_or(false));
                if reused && !known.contains_key(&key) {
                    return Err(PyValueError::new_err(
                        "Reuse refers to an unknown artifact version",
                    ));
                }
                if !reused && known.contains_key(&key) {
                    return Err(PyValueError::new_err("Duplicate artifact version"));
                }
                let assigned = aliases(row)?;
                for (other_key, other) in &known {
                    if other_key.0 == key.0 && other_key != &key {
                        let remaining = aliases(other.bind(py))?
                            .difference(&assigned)
                            .cloned()
                            .collect();
                        set_aliases(other.bind(py), remaining)?;
                    }
                }
                if reused {
                    let previous = known.get(&key).unwrap();
                    let mut merged = aliases(previous.bind(py))?;
                    merged.extend(aliases(row)?);
                    set_aliases(previous.bind(py), merged)?;
                } else {
                    let owned = row.copy()?.unbind();
                    records.append(owned.bind(py))?;
                    known.insert(key, owned);
                }
                Ok(())
            })();
            if let Err(error) = decoded {
                diagnostics.push(format!(
                    "Skipping corrupted artifact index line {} in {}: {}",
                    index + 1,
                    path.display(),
                    error
                ));
            }
        }
        Ok((records.unbind(), diagnostics))
    }

    fn append(&self, py: Python<'_>, record: &Bound<'_, PyDict>) -> PyResult<()> {
        let text = dumps(py, record.as_any())? + "\n";
        // Replacing the complete index prevents a partial append from exposing
        // a version, and keeps readers on a consistent index snapshot.
        let index = self.root.join("index.jsonl");
        py.detach(|| -> std::io::Result<()> {
            let mut target = tempfile::NamedTempFile::new_in(&self.root)?;
            if index.exists() {
                let mut source = File::open(&index)?;
                std::io::copy(&mut source, &mut target)?;
                let length = target.as_file().metadata()?.len();
                if length > 0 {
                    use std::io::{Seek, SeekFrom};
                    target.as_file_mut().seek(SeekFrom::End(-1))?;
                    let mut last = [0];
                    target.as_file_mut().read_exact(&mut last)?;
                    target.as_file_mut().seek(SeekFrom::End(0))?;
                    if last[0] != b'\n' {
                        target.write_all(b"\n")?;
                    }
                }
            }
            target.write_all(text.as_bytes())?;
            target.persist(&index).map_err(|e| e.error)?;
            Ok(())
        })?;
        Ok(())
    }
}

#[pyfunction]
fn artifact_copy(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    target: &Bound<'_, PyAny>,
    entries: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let source = path_value(source, false)?;
    let target = path_value(target, false)?;
    let entries = entries
        .try_iter()?
        .map(|v| path_value(&v?, false))
        .collect::<PyResult<Vec<_>>>()?;
    for path in &entries {
        validate(path, "entry path")?;
    }
    py.detach(|| -> std::io::Result<()> {
        fs::create_dir_all(&target)?;
        for path in entries {
            let from = source.join(&path);
            let to = target.join(path);
            // Refuse destination symlinks rather than copying out of root.
            let mut ancestor = to.as_path();
            while ancestor != target {
                if fs::symlink_metadata(ancestor).is_ok_and(|m| m.file_type().is_symlink()) {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Artifact destination contains a symlink",
                    ));
                }
                ancestor = ancestor.parent().unwrap_or(&target);
            }
            fs::create_dir_all(to.parent().unwrap())?;
            if from.exists() {
                copy_file(&from, &to)?;
            }
        }
        Ok(())
    })?;
    Ok(())
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeArtifactStore>()?;
    module.add_function(wrap_pyfunction!(artifact_file_digest, module)?)?;
    module.add_function(wrap_pyfunction!(artifact_validate_path, module)?)?;
    module.add_function(wrap_pyfunction!(artifact_copy, module)?)?;
    module.add_function(wrap_pyfunction!(artifact_file_entry, module)?)?;
    module.add_function(wrap_pyfunction!(artifact_directory_entries, module)?)?;
    module.add_function(wrap_pyfunction!(artifact_append_lineage, module)?)
}
