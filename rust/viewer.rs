use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use axum::extract::{Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::{Json, Router};
use parking_lot::{Mutex, RwLock, RwLockWriteGuard};
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use tower_http::services::ServeDir;

use crate::history::{segment_paths, visit_rows_from};
use crate::viewer_json;

type Result<T> = std::result::Result<T, Error>;

fn finite_number(value: &Value) -> Option<f64> {
    match value {
        Value::Bool(value) => Some(f64::from(u8::from(*value))),
        value => value.as_f64().filter(|number| number.is_finite()),
    }
}

#[derive(Debug)]
struct Error(StatusCode, String);

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.0, Json(json!({"error": self.1}))).into_response()
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Self(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

impl From<serde_json::Error> for Error {
    fn from(error: serde_json::Error) -> Self {
        Self(StatusCode::BAD_REQUEST, error.to_string())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error(StatusCode::BAD_REQUEST, message.into())
}

#[derive(Clone, Serialize)]
struct LogFile {
    name: String,
    kind: String,
    stream: Option<String>,
    rank: u64,
}

#[derive(Clone, Serialize)]
struct Run {
    id: String,
    name: String,
    project: String,
    bytes: u64,
    modified: f64,
    files: Vec<LogFile>,
    #[serde(skip)]
    directory: PathBuf,
    #[serde(skip)]
    root: PathBuf,
}

struct Viewer {
    root: PathBuf,
    hosts: Vec<String>,
    runs: Mutex<HashMap<String, Run>>,
    logs: RwLock<HashMap<PathBuf, Mutex<Option<CachedLog>>>>,
    stopping: Arc<AtomicBool>,
}

mod index;
use index::*;
mod directories;
use directories::RootQuery;

fn log_file(name: &str) -> Option<LogFile> {
    let stem = name.strip_suffix(".jsonl")?;
    let mut parts: Vec<&str> = stem.split('.').collect();
    let kind = parts.first()?;
    if !["metrics", "spans"].contains(kind) {
        return None;
    }
    let mut rank = 0;
    if parts.len() > 1 {
        if let Some(number) = parts
            .last()?
            .strip_prefix("rank")
            .and_then(|rank| rank.parse::<u64>().ok())
        {
            rank = number;
            parts.pop();
        }
    }
    if parts.len() > 2 {
        return None;
    }
    Some(LogFile {
        name: name.into(),
        kind: parts[0].into(),
        stream: parts.get(1).map(|stream| (*stream).into()),
        rank,
    })
}

impl Viewer {
    fn discover(&self, root: &Path) -> Result<Vec<Run>> {
        let mut found = Vec::new();
        self.walk(root, root, 0, &mut found)?;
        found.sort_by(|left, right| {
            right
                .modified
                .total_cmp(&left.modified)
                .then_with(|| left.id.cmp(&right.id))
        });
        *self.runs.lock() = found
            .iter()
            .map(|run| (run.id.clone(), run.clone()))
            .collect();
        Ok(found)
    }

    fn walk(
        &self,
        root: &Path,
        directory: &Path,
        depth: usize,
        output: &mut Vec<Run>,
    ) -> Result<()> {
        let mut files = Vec::new();
        let mut directories = Vec::new();
        let mut bytes = 0;
        let mut modified: f64 = 0.0;
        for entry in directory.read_dir()? {
            if self.stopping.load(Ordering::Relaxed) {
                return Err(
                    io::Error::new(io::ErrorKind::Interrupted, "Viewer is shutting down").into(),
                );
            }
            let entry = entry?;
            let kind = entry.file_type()?;
            let name = entry.file_name();
            let text = name.to_string_lossy();
            if kind.is_file() {
                if let Some(file) = log_file(&text) {
                    for segment in segment_paths(&entry.path())? {
                        let metadata = segment.metadata()?;
                        bytes += metadata.len();
                        modified = modified.max(
                            metadata
                                .modified()?
                                .duration_since(UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs_f64(),
                        );
                    }
                    files.push(file);
                }
            } else if kind.is_dir()
                && !text.starts_with('.')
                && !text.ends_with(".parts")
                && text != "artifacts"
            {
                directories.push(entry.path());
            }
        }
        if !files.is_empty() {
            files.sort_by(|left, right| left.name.cmp(&right.name));
            let relative = directory
                .strip_prefix(root)
                .map_err(|error| invalid(error.to_string()))?;
            let project = relative.components().next().map_or_else(
                || "local".into(),
                |part| part.as_os_str().to_string_lossy().into_owned(),
            );
            let name = directory
                .file_name()
                .unwrap_or(directory.as_os_str())
                .to_string_lossy()
                .into_owned();
            let id = format!(
                "{:x}",
                Sha256::digest(directory.as_os_str().as_encoded_bytes())
            );
            output.push(Run {
                id,
                name,
                project,
                bytes,
                modified,
                files,
                directory: directory.to_path_buf(),
                root: root.to_path_buf(),
            });
        } else if depth < 8 {
            for child in directories {
                self.walk(root, &child, depth + 1, output)?;
            }
        }
        Ok(())
    }

    fn run(&self, root: &Path, id: &str) -> Result<Run> {
        if let Some(run) = self
            .runs
            .lock()
            .get(id)
            .filter(|run| run.root == root)
            .cloned()
        {
            return Ok(run);
        }
        // Use this discovery's result: another page may replace the shared cache.
        self.discover(root)?
            .into_iter()
            .find(|run| run.id == id)
            .ok_or_else(|| Error(StatusCode::NOT_FOUND, "Run not found".into()))
    }

    fn file(&self, run: &Run, wanted: Option<&str>) -> Result<(PathBuf, bool)> {
        let file = match wanted {
            Some(name) => run.files.iter().find(|file| file.name == name),
            None => run
                .files
                .iter()
                .find(|file| file.name == "metrics.jsonl")
                .or_else(|| run.files.iter().find(|file| file.kind == "metrics")),
        }
        .ok_or_else(|| {
            Error(
                StatusCode::NOT_FOUND,
                "Log source not found in this run".into(),
            )
        })?;
        let path = run.directory.join(&file.name).canonicalize()?;
        if !path.starts_with(&run.root) {
            return Err(Error(
                StatusCode::FORBIDDEN,
                "Log source is outside the configured root".into(),
            ));
        }
        for part in segment_paths(&path)? {
            if !part.canonicalize()?.starts_with(&run.root) {
                return Err(Error(
                    StatusCode::FORBIDDEN,
                    "Log segment is outside the configured root".into(),
                ));
            }
        }
        Ok((path, file.kind == "spans"))
    }

    fn with_log<T>(
        &self,
        path: &Path,
        spans: bool,
        action: impl FnOnce(&CachedLog) -> Result<T>,
    ) -> Result<T> {
        // The registry read guard pins indices until their query finishes.
        let mut logs = self.logs.read();
        loop {
            if let Some(slot) = logs.get(path) {
                let mut cached = slot.lock();
                if let Some(log) = cached.as_mut() {
                    if let Err(error) = log.refresh(path, spans) {
                        *cached = None;
                        return Err(error);
                    }
                    return action(log);
                }
            }
            drop(logs);
            let mut logs_write = self.logs.write();
            logs_write.retain(|_, slot| slot.get_mut().is_some());
            if !logs_write.contains_key(path) {
                if logs_write.len() >= 32 {
                    if let Some(old) = logs_write.keys().next().cloned() {
                        logs_write.remove(&old);
                    }
                }
                logs_write.insert(
                    path.to_path_buf(),
                    Mutex::new(Some(CachedLog::new(self.stopping.clone())?)),
                );
            }
            logs = RwLockWriteGuard::downgrade(logs_write);
        }
    }
}

#[derive(Deserialize)]
struct Selection {
    run: String,
    file: Option<String>,
    root: Option<String>,
}

#[derive(Deserialize)]
struct HistoryQuery {
    run: String,
    file: Option<String>,
    metrics: String,
    axis: Option<String>,
    max_points: Option<usize>,
    from_step: Option<String>,
    to_step: Option<String>,
    root: Option<String>,
}

#[derive(Deserialize)]
struct RecordsQuery {
    run: String,
    file: Option<String>,
    limit: Option<usize>,
    before: Option<i64>,
    from_step: Option<String>,
    to_step: Option<String>,
    root: Option<String>,
}

async fn blocking<T: Send + 'static>(
    action: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<Json<T>> {
    tokio::task::spawn_blocking(action)
        .await
        .map_err(|error| {
            Error(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Query worker failed: {error}"),
            )
        })?
        .map(Json)
}

async fn runs(
    State(viewer): State<Arc<Viewer>>,
    Query(query): Query<RootQuery>,
) -> Result<Json<Vec<Run>>> {
    blocking(move || viewer.discover(&viewer.resolve_root(query.root.as_deref())?)).await
}

fn sidecars(run: &Run, base: &str) -> Result<Value> {
    let mut result = Map::new();
    let mut bytes = 0;
    for entry in run.directory.read_dir()? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if entry.file_type()?.is_file() && name.starts_with(base) && name.ends_with(".json") {
            bytes += entry.metadata()?.len();
            if bytes > 16 * 1024 * 1024 {
                return Err(invalid(format!(
                    "{base} metadata exceeds the viewer's 16 MiB response budget"
                )));
            }
            let parsed = viewer_json::parse(&std::fs::read(entry.path())?)?;
            result.insert(name, viewer_json::browser_value(parsed.value));
        }
    }
    Ok(Value::Object(result))
}

async fn details(
    State(viewer): State<Arc<Viewer>>,
    Query(selection): Query<Selection>,
) -> Result<Json<Value>> {
    blocking(move || {
        let root = viewer.resolve_root(selection.root.as_deref())?;
        let run = viewer.run(&root, &selection.run)?;
        let config = sidecars(&run, "config")?;
        let summary = sidecars(&run, "summary")?;
        if !run.files.iter().any(|file| file.kind == "metrics") && selection.file.is_none() {
            return Ok(json!({"run": run, "config": config, "summary": summary, "metrics": [], "rows": 0, "warnings": []}));
        }
        let (path, spans) = viewer.file(&run, selection.file.as_deref())?;
        viewer.with_log(&path, spans, |log| {
            let rows: u64 = log.db.query_row("SELECT count(*) FROM records", [], |row| row.get(0))?;
            Ok(json!({"run": run, "config": config, "summary": summary, "metrics": log.metrics, "rows": rows, "warnings": log.warnings()}))
        })
    }).await
}

#[derive(Clone)]
struct Point {
    at: u64,
    x: f64,
    ys: Vec<Option<f64>>,
}

struct Bucket {
    points: BTreeMap<u64, Point>,
    extrema: Vec<Option<(Point, Point)>>,
    missing: Vec<Option<Point>>,
    last: Option<Point>,
}

impl Bucket {
    fn new(metrics: usize) -> Self {
        Self {
            points: BTreeMap::new(),
            extrema: vec![None; metrics],
            missing: vec![None; metrics],
            last: None,
        }
    }

    fn push(&mut self, point: Point) {
        if self.points.is_empty() {
            self.points.insert(point.at, point.clone());
        }
        for (at, value) in point.ys.iter().enumerate() {
            if let Some(value) = value {
                if let Some((min, max)) = self.extrema[at].as_mut() {
                    if *value < min.ys[at].unwrap() {
                        *min = point.clone();
                    }
                    if *value > max.ys[at].unwrap() {
                        *max = point.clone();
                    }
                } else {
                    self.extrema[at] = Some((point.clone(), point.clone()));
                }
            } else if self.missing[at].is_none() {
                self.missing[at] = Some(point.clone());
            }
        }
        self.last = Some(point);
    }

    fn finish(mut self) -> Vec<Point> {
        if let Some(last) = self.last {
            self.points.insert(last.at, last);
        }
        for (min, max) in self.extrema.into_iter().flatten() {
            self.points.insert(min.at, min);
            self.points.insert(max.at, max);
        }
        for missing in self.missing.into_iter().flatten() {
            self.points.insert(missing.at, missing);
        }
        self.points.into_values().collect()
    }
}

fn chart(log: &CachedLog, query: &HistoryQuery) -> Result<Value> {
    let metrics: Vec<String> = serde_json::from_str(&query.metrics)?;
    if metrics.len() > 12 || metrics.iter().any(|metric| metric.len() > 4096) {
        return Err(invalid("Select at most twelve metrics"));
    }
    let max_points = query.max_points.unwrap_or(1000);
    if !(64..=10_000).contains(&max_points) {
        return Err(invalid("max_points must be between 64 and 10000"));
    }
    let axis = query.axis.as_deref().unwrap_or("step");
    if !["step", "time"].contains(&axis) {
        return Err(invalid("Unknown chart axis"));
    }
    let low = bound(query.from_step.as_deref())?;
    let high = bound(query.to_step.as_deref())?;
    if low
        .as_ref()
        .zip(high.as_ref())
        .is_some_and(|(low, high)| low >= high)
    {
        return Err(invalid("from_step must be less than to_step"));
    }
    let total: u64 = if low.is_none() && high.is_none() {
        log.db
            .query_row("SELECT count(*) FROM records", [], |row| row.get(0))?
    } else {
        log.db.query_row(
            "SELECT count(*) FROM records WHERE sort>=?1 AND sort<?2",
            params![low.as_deref().unwrap_or(""), high.as_deref().unwrap_or("2")],
            |row| row.get(0),
        )?
    };
    let sql = if axis == "time" {
        "SELECT data FROM records WHERE sort>=?1 AND sort<?2 ORDER BY time,sort"
    } else {
        "SELECT data FROM records WHERE sort>=?1 AND sort<?2 ORDER BY sort"
    };
    let mut statement = log.db.prepare(sql)?;
    let mut rows = statement.query(params![
        low.as_deref().unwrap_or(""),
        high.as_deref().unwrap_or("2")
    ])?;
    let buckets = (max_points / (3 * metrics.len() + 2)).max(1) as u64;
    let mut bucket_at = 0;
    let mut bucket = Bucket::new(metrics.len());
    let mut points = Vec::new();
    let mut nonfinite = vec![0u64; metrics.len()];
    let mut bad_axis = 0;
    let mut at = 0;
    while let Some(row) = rows.next()? {
        let raw: String = row.get(0)?;
        let value: Value = serde_json::from_str(&raw)?;
        let next = at * buckets / total.max(1);
        if next != bucket_at {
            points.extend(bucket.finish());
            bucket = Bucket::new(metrics.len());
            bucket_at = next;
        }
        let field = if axis == "step" { "_step" } else { "_time" };
        let x = value.get(field).and_then(finite_number);
        let exact_axis = axis == "time"
            || value.get(field).is_some_and(Value::is_boolean)
            || value
                .get(field)
                .and_then(Value::as_i64)
                .is_some_and(|step| step.unsigned_abs() <= 9_007_199_254_740_991);
        if let Some(x) = x.filter(|_| exact_axis) {
            let ys = metrics
                .iter()
                .enumerate()
                .map(|(index, metric)| {
                    let metric = value.get(metric)?;
                    let number = finite_number(metric);
                    if metric.get("$et.nonfinite").is_some()
                        || (metric.is_number() && number.is_none())
                    {
                        nonfinite[index] += 1;
                    }
                    number
                })
                .collect();
            bucket.push(Point { at, x, ys });
        } else {
            bad_axis += 1;
        }
        at += 1;
    }
    points.extend(bucket.finish());
    let series: Vec<Value> = metrics.iter().enumerate().map(|(at, metric)| json!({
        "metric": metric, "values": points.iter().map(|point| point.ys[at]).collect::<Vec<_>>(), "nonfinite": nonfinite[at],
    })).collect();
    let mut warnings = log.warnings();
    if bad_axis > 0 {
        warnings.push(format!("{bad_axis} records have an unrepresentable chart axis; inspect Records for exact values."));
    }
    Ok(
        json!({"x": points.iter().map(|point| point.x).collect::<Vec<_>>(), "series": series, "total": total, "warnings": warnings}),
    )
}

async fn history(
    State(viewer): State<Arc<Viewer>>,
    Query(query): Query<HistoryQuery>,
) -> Result<Json<Value>> {
    blocking(move || {
        let root = viewer.resolve_root(query.root.as_deref())?;
        let run = viewer.run(&root, &query.run)?;
        let (path, spans) = viewer.file(&run, query.file.as_deref())?;
        if spans {
            return Err(invalid("Charts require a metric source"));
        }
        viewer.with_log(&path, false, |log| chart(log, &query))
    })
    .await
}

async fn records(
    State(viewer): State<Arc<Viewer>>,
    Query(query): Query<RecordsQuery>,
) -> Result<Json<Value>> {
    blocking(move || {
        let limit = query.limit.unwrap_or(100);
        if !(1..=1000).contains(&limit) {
            return Err(invalid("limit must be between 1 and 1000"));
        }
        let low = bound(query.from_step.as_deref())?;
        let high = bound(query.to_step.as_deref())?;
        if low
            .as_ref()
            .zip(high.as_ref())
            .is_some_and(|(low, high)| low >= high)
        {
            return Err(invalid("from_step must be less than to_step"));
        }
        let root = viewer.resolve_root(query.root.as_deref())?;
        let run = viewer.run(&root, &query.run)?;
        let (path, spans) = viewer.file(&run, query.file.as_deref())?;
        viewer.with_log(&path, spans, |log| {
            let ranged = low.is_some() || high.is_some();
            let sql = if ranged {
                "SELECT data,ordinal FROM records WHERE ordinal<?1
                 AND sort>=?2 AND sort<?3 ORDER BY ordinal DESC LIMIT ?4"
            } else {
                "SELECT data,ordinal FROM records WHERE ordinal<?1
                 ORDER BY ordinal DESC LIMIT ?2"
            };
            let mut statement = log.db.prepare(sql)?;
            let mut rows = if ranged {
                statement.query(params![
                    query.before.unwrap_or(i64::MAX),
                    low.as_deref().unwrap_or(""),
                    high.as_deref().unwrap_or("2"),
                    limit + 1
                ])
            } else {
                statement.query(params![query.before.unwrap_or(i64::MAX), limit + 1])
            }?;
            let mut records = Vec::new();
            let mut last = 0;
            let mut next = None;
            let mut bytes = 0;
            while let Some(row) = rows.next()? {
                if records.len() == limit {
                    next = Some(last.to_string());
                    break;
                }
                let raw: String = row.get(0)?;
                last = row.get::<_, i64>(1)?;
                let value = viewer_json::browser_value(serde_json::from_str(&raw)?);
                bytes += serde_json::to_vec(&value)?.len();
                if bytes > 16 * 1024 * 1024 {
                    return Err(invalid(
                        "Records exceed the 16 MiB response budget; reduce limit or narrow the range",
                    ));
                }
                records.push(value);
            }
            Ok(json!({"records": records, "next": next, "warnings": log.warnings()}))
        })
    })
    .await
}

async fn boundary(
    State(viewer): State<Arc<Viewer>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let host = request
        .headers()
        .get(header::HOST)
        .and_then(|host| host.to_str().ok());
    if !host.is_some_and(|host| viewer.hosts.iter().any(|allowed| allowed == host)) {
        return Error(StatusCode::FORBIDDEN, "Unexpected request host".into()).into_response();
    }
    if let Some(origin) = request.headers().get(header::ORIGIN) {
        if !origin.to_str().ok().is_some_and(|origin| {
            viewer
                .hosts
                .iter()
                .any(|host| origin == format!("http://{host}"))
        }) {
            return Error(
                StatusCode::FORBIDDEN,
                "Cross-origin requests are not permitted".into(),
            )
            .into_response();
        }
    }
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"));
    response
}

fn router(viewer: Arc<Viewer>, ui: PathBuf) -> Router {
    Router::new()
        .route("/api/root", axum::routing::get(directories::location))
        .route("/api/directories", axum::routing::get(directories::browse))
        .route("/api/runs", axum::routing::get(runs))
        .route("/api/run", axum::routing::get(details))
        .route("/api/history", axum::routing::get(history))
        .route("/api/records", axum::routing::get(records))
        .fallback_service(ServeDir::new(ui))
        .layer(middleware::from_fn_with_state(viewer.clone(), boundary))
        .with_state(viewer)
}

#[pyfunction]
#[pyo3(signature = (root, ui, host="127.0.0.1", port=9110))]
fn serve(py: Python<'_>, root: PathBuf, ui: PathBuf, host: &str, port: u16) -> PyResult<()> {
    let ip: IpAddr = host
        .parse()
        .map_err(|_| PyValueError::new_err("host must be a loopback IP address"))?;
    if !ip.is_loopback() {
        return Err(PyValueError::new_err("The read-only local viewer must bind to a loopback address. Use an SSH tunnel for remote access."));
    }
    let root = root
        .canonicalize()
        .map_err(|error| PyOSError::new_err(error.to_string()))?;
    if !root.is_dir() {
        return Err(PyValueError::new_err("root must be a directory"));
    }
    if !ui.join("index.html").is_file() {
        return Err(PyOSError::new_err("Web assets are missing. Build them with npm --prefix web ci && npm --prefix web run build."));
    }
    let address = SocketAddr::new(ip, port);
    let viewer = Arc::new(Viewer {
        root,
        hosts: vec![address.to_string(), format!("localhost:{port}")],
        runs: Mutex::new(HashMap::new()),
        logs: RwLock::new(HashMap::new()),
        stopping: Arc::new(AtomicBool::new(false)),
    });
    py.detach(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind(address).await?;
            eprintln!("Experiment viewer: http://{address}");
            let stopping = viewer.stopping.clone();
            axum::serve(listener, router(viewer, ui))
                .with_graceful_shutdown(async move {
                    if let Err(error) = tokio::signal::ctrl_c().await {
                        eprintln!("Could not install viewer shutdown handler: {error}");
                    }
                    stopping.store(true, Ordering::Relaxed);
                })
                .await
        })
    })
    .map_err(|error: io::Error| PyOSError::new_err(error.to_string()))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(serve, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;
