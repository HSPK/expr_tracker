use super::{
    finite_number, invalid, io, params, segment_paths, viewer_json, visit_rows_from, Arc,
    AtomicBool, BTreeSet, Connection, Error, Map, Ordering, Path, PathBuf, Result, Value,
    UNIX_EPOCH,
};
use rusqlite::OptionalExtension;

#[derive(Clone, PartialEq)]
pub(super) struct Fingerprint {
    pub(super) path: PathBuf,
    pub(super) size: u64,
    pub(super) identity: u64,
    pub(super) modified: u128,
}

pub(super) struct CachedLog {
    pub(super) db: Connection,
    pub(super) prefix: u64,
    pub(super) ordinal: i64,
    pub(super) fingerprint: Vec<Fingerprint>,
    pub(super) metrics: BTreeSet<String>,
    pub(super) corrupt: u64,
    pub(super) surrogates: bool,
    pub(super) stopping: Arc<AtomicBool>,
}

impl CachedLog {
    pub(super) fn new(stopping: Arc<AtomicBool>) -> Result<Self> {
        let db = Connection::open("")?;
        let cancelled = stopping.clone();
        db.progress_handler(10_000, Some(move || cancelled.load(Ordering::Relaxed)));
        db.execute_batch(
            "PRAGMA cache_size=-8192;
             PRAGMA temp_store=FILE;
             CREATE TABLE records (
                 step TEXT PRIMARY KEY, sort TEXT NOT NULL,
                 ordinal INTEGER NOT NULL, time REAL, data TEXT NOT NULL
             );
             CREATE INDEX records_sort ON records(sort);
             CREATE INDEX records_time ON records(time);
             CREATE INDEX records_ordinal ON records(ordinal);",
        )?;
        Ok(Self {
            db,
            prefix: 0,
            ordinal: 0,
            fingerprint: Vec::new(),
            metrics: BTreeSet::new(),
            corrupt: 0,
            surrogates: false,
            stopping,
        })
    }

    pub(super) fn warnings(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.corrupt > 0 {
            warnings.push(format!("Skipped {} malformed records.", self.corrupt));
        }
        if self.surrogates {
            warnings.push("Unpaired Unicode surrogates are displayed as escapes.".into());
        }
        warnings
    }

    pub(super) fn refresh(&mut self, path: &Path, spans: bool) -> Result<()> {
        let current = fingerprint(path)?;
        if current == self.fingerprint {
            return Ok(());
        }
        let append_only = current.len() >= self.fingerprint.len()
            && self
                .fingerprint
                .iter()
                .zip(&current)
                .enumerate()
                .all(|(at, (old, new))| {
                    old.path == new.path
                        && old.identity == new.identity
                        && new.size >= old.size
                        && (at + 1 == self.fingerprint.len()
                            || (old.size == new.size && old.modified == new.modified))
                        && (old.size != new.size || old.modified == new.modified)
                });
        if !append_only {
            *self = Self::new(self.stopping.clone())?;
        }
        let transaction = self.db.transaction()?;
        let mut insert = transaction.prepare(
            "INSERT INTO records(step,sort,ordinal,time,data) VALUES (?1,?2,?3,?4,?5)
             ON CONFLICT(step) DO UPDATE SET
                 ordinal=excluded.ordinal,time=excluded.time,data=excluded.data",
        )?;
        let mut previous = transaction.prepare("SELECT data FROM records WHERE step=?1")?;
        let mut failure: Option<Error> = None;
        let prefix = visit_rows_from(path, self.prefix, |raw| {
            if self.stopping.load(Ordering::Relaxed) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "Viewer is shutting down",
                ));
            }
            let parsed = match viewer_json::parse(raw) {
                Ok(value) => value,
                Err(_) => {
                    self.corrupt += 1;
                    return Ok(());
                }
            };
            self.surrogates |= parsed.escaped_surrogates;
            let Value::Object(mut values) = parsed.value else {
                self.corrupt += 1;
                return Ok(());
            };
            let (step, order) = if let Some(step) = step_key(values.get("_step")) {
                step
            } else if spans && !values.contains_key("_step") {
                step_key(Some(&Value::from(0))).expect("Zero is an integer step")
            } else {
                self.corrupt += 1;
                return Ok(());
            };
            let key = if spans {
                self.ordinal.to_string()
            } else {
                step
            };
            let sort = order;
            for (key, value) in &values {
                if !["_step", "_time"].contains(&key.as_str())
                    && (value.is_number()
                        || value.is_boolean()
                        || value.get("$et.nonfinite").is_some())
                {
                    self.metrics.insert(key.clone());
                }
            }
            let result = (|| -> Result<()> {
                if !spans {
                    let old: Option<String> =
                        previous.query_row([&key], |row| row.get(0)).optional()?;
                    if let Some(old) = old {
                        let mut combined: Map<String, Value> = serde_json::from_str(&old)?;
                        combined.extend(values);
                        values = combined;
                    }
                }
                let time = values.get("_time").and_then(finite_number);
                insert.execute(params![
                    key,
                    sort,
                    self.ordinal,
                    time,
                    serde_json::to_string(&values)?
                ])?;
                Ok(())
            })();
            if let Err(error) = result {
                failure = Some(error);
                return Err(io::Error::other("Could not index log records"));
            }
            self.ordinal += 1;
            Ok(())
        });
        drop(insert);
        drop(previous);
        if let Some(error) = failure {
            return Err(error);
        }
        self.prefix = prefix?;
        transaction.commit()?;
        self.fingerprint = current;
        Ok(())
    }
}

pub(super) fn fingerprint(path: &Path) -> Result<Vec<Fingerprint>> {
    segment_paths(path)?
        .into_iter()
        .filter(|path| path.exists())
        .map(|path| {
            let metadata = path.metadata()?;
            #[cfg(unix)]
            let identity = {
                use std::os::unix::fs::MetadataExt;
                metadata.ino() ^ metadata.dev().rotate_left(32)
            };
            #[cfg(not(unix))]
            let identity = metadata
                .created()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |time| time.as_nanos() as u64);
            Ok(Fingerprint {
                path,
                size: metadata.len(),
                identity,
                modified: metadata
                    .modified()?
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos(),
            })
        })
        .collect()
}

pub(super) fn step_key(value: Option<&Value>) -> Option<(String, String)> {
    let value = value?;
    let number = match value {
        Value::Number(number) => number.to_string(),
        Value::Bool(value) => u8::from(*value).to_string(),
        _ => return None,
    };
    let negative = number.starts_with('-');
    let digits = number.strip_prefix('-').unwrap_or(&number);
    if digits.is_empty() || !digits.bytes().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some(("0".into(), format!("1{:020}0", 1)));
    }
    if negative {
        let inverse: String = digits
            .bytes()
            .map(|digit| char::from(b'9' - (digit - b'0')))
            .collect();
        Some((
            format!("-{digits}"),
            format!("0{:020}{inverse}", u64::MAX - digits.len() as u64),
        ))
    } else {
        Some((digits.into(), format!("1{:020}{digits}", digits.len())))
    }
}

pub(super) fn bound(value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| {
            if value.len() > 4096 {
                return Err(invalid("Step bound is too long"));
            }
            let number: Value = serde_json::from_str(value)?;
            if !number.is_number() {
                return Err(invalid("Step bounds must be integers"));
            }
            step_key(Some(&number))
                .map(|(_, sort)| sort)
                .ok_or_else(|| invalid("Step bounds must be integers"))
        })
        .transpose()
}
