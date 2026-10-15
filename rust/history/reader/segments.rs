use super::{
    Arc, AtomicUsize, BTreeMap, BufReader, File, HashSet, JsonObject, Mutex, Ordering, Path,
    PathBuf, SeekFrom, Step, VecDeque, DATA_OPENS,
};
use std::fs;
use std::io::{self, BufRead, Read, Seek};

pub(super) fn open_data(path: &Path) -> io::Result<File> {
    DATA_OPENS.fetch_add(1, Ordering::Relaxed);
    File::open(path)
}

pub fn segment_paths(path: &Path) -> io::Result<Vec<PathBuf>> {
    let parts = PathBuf::from(format!("{}.parts", path.display()));
    let mut numbered = Vec::new();
    match fs::read_dir(parts) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                let p = entry.path();
                if p.extension().and_then(|x| x.to_str()) == Some("jsonl") {
                    if let Some(number) = p
                        .file_stem()
                        .and_then(|x| x.to_str())
                        .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                        .and_then(|s| s.parse::<u64>().ok())
                    {
                        numbered.push((number, p));
                    }
                }
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => (),
        Err(e) => return Err(e),
    }
    numbered.sort_by_key(|(n, _)| *n);
    let mut result = vec![path.to_owned()];
    result.extend(numbered.into_iter().map(|(_, p)| p));
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct RawRow {
    pub offset: u64,
    pub bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
pub struct SegmentedReader {
    pub path: PathBuf,
    pub segments: Vec<(PathBuf, u64)>,
    pub size: u64,
    pub(super) limit: u64,
    pub(super) hints: Option<(Vec<(Step, usize, u64)>, bool)>,
    pub(super) scanned: Arc<AtomicUsize>,
    pub(super) integer_limit: usize,
    pub(super) warnings: Option<Arc<Mutex<Vec<String>>>>,
}
impl SegmentedReader {
    /// Pin the segment list and byte lengths once; concurrent appends are invisible.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut segments = Vec::new();
        for p in segment_paths(path)? {
            match fs::metadata(&p) {
                Ok(meta) => segments.push((p, meta.len())),
                Err(e) if e.kind() == io::ErrorKind::NotFound => (),
                Err(e) => return Err(e),
            }
        }
        let size = segments.iter().map(|(_, n)| n).sum();
        Ok(Self {
            path: path.into(),
            segments,
            size,
            limit: size,
            hints: None,
            scanned: Arc::new(AtomicUsize::new(0)),
            integer_limit: 0,
            warnings: None,
        })
    }
    pub fn snapshot(&self, end: Option<u64>) -> Self {
        let mut result = self.clone();
        result.limit = end.unwrap_or(self.limit).min(self.limit);
        result
    }
    pub fn limit(&self) -> u64 {
        self.limit
    }
    pub(super) fn parse_row(&self, bytes: &[u8]) -> Option<JsonObject> {
        if super::super::json::exceeds_integer_limit(bytes, self.integer_limit) {
            self.warn(format!(
                "Skipping corrupted line in {}: integer string conversion limit exceeded",
                self.path.display()
            ));
            return None;
        }
        let parsed = parse_row(bytes);
        if parsed.is_none() && super::super::json::parse_compatible_json(bytes).is_err() {
            self.warn(format!(
                "Skipping corrupted line in {}: invalid JSON",
                self.path.display()
            ));
        }
        parsed
    }
    pub(super) fn validated_step(&self, bytes: &[u8]) -> Option<Step> {
        if super::super::json::exceeds_integer_limit(bytes, self.integer_limit) {
            self.warn(format!(
                "Skipping corrupted line in {}: integer string conversion limit exceeded",
                self.path.display()
            ));
            return None;
        }
        let step = std::str::from_utf8(bytes)
            .ok()
            .and_then(JsonObject::validated_step);
        if step.is_none() {
            self.parse_row(bytes);
        }
        step
    }
    pub(super) fn warn(&self, message: String) {
        if let Some(warnings) = &self.warnings {
            let mut warnings = warnings.lock();
            if warnings.len() < 1024 && warnings.last() != Some(&message) {
                warnings.push(message);
            }
        }
    }
    pub(in crate::history) fn with_hints(
        mut self,
        index: Vec<(Step, usize, u64)>,
        merge: bool,
    ) -> Self {
        self.hints = Some((
            index.into_iter().filter(|e| e.2 < self.limit).collect(),
            merge,
        ));
        self
    }
    /// Visit physical rows in bounded memory. Unterminated rows are allowed only
    /// at a captured physical EOF, never at a caller's crossing byte boundary.
    pub fn visit(
        &self,
        start: u64,
        unterminated: bool,
        mut visitor: impl FnMut(RawRow) -> io::Result<()>,
    ) -> io::Result<()> {
        self.visit_until(start, unterminated, |row| visitor(row).map(|_| true))
    }
    pub(super) fn visit_until(
        &self,
        start: u64,
        unterminated: bool,
        mut visitor: impl FnMut(RawRow) -> io::Result<bool>,
    ) -> io::Result<()> {
        let mut base = 0;
        for (path, length) in &self.segments {
            if base >= self.limit {
                break;
            }
            let local_end = (*length).min(self.limit - base);
            let local_start = start.saturating_sub(base);
            if local_start < local_end {
                let mut file = open_data(path)?;
                file.seek(SeekFrom::Start(local_start))?;
                let mut reader =
                    BufReader::with_capacity(256 * 1024, file.take(local_end - local_start));
                let mut offset = local_start;
                loop {
                    let mut bytes = Vec::new();
                    let count = reader.read_until(b'\n', &mut bytes)?;
                    if count == 0 {
                        break;
                    }
                    let complete = bytes.last() == Some(&b'\n');
                    if !complete && !(unterminated && local_end == *length) {
                        break;
                    }
                    if bytes.iter().any(|b| !b.is_ascii_whitespace()) {
                        self.scanned.fetch_add(1, Ordering::Relaxed);
                        if !visitor(RawRow {
                            offset: base + offset,
                            bytes,
                        })? {
                            return Ok(());
                        }
                    }
                    offset += count as u64;
                }
            }
            base += length;
        }
        Ok(())
    }
    pub fn rows(&self, unterminated: bool) -> io::Result<Vec<RawRow>> {
        let mut rows = Vec::new();
        self.visit(0, unterminated, |row| {
            rows.push(row);
            Ok(())
        })?;
        Ok(rows)
    }
    pub fn reverse(&self, mut visitor: impl FnMut(RawRow) -> io::Result<bool>) -> io::Result<()> {
        let mut base: u64 = self.segments.iter().map(|(_, n)| n).sum();
        for (path, length) in self.segments.iter().rev() {
            base -= length;
            if base >= self.limit {
                continue;
            }
            let end = (*length).min(self.limit - base);
            if end == 0 {
                continue;
            }
            let mut file = open_data(path)?;
            file.seek(SeekFrom::Start(end - 1))?;
            let mut last = [0];
            file.read_exact(&mut last)?;
            let mut skip_partial = end < *length && last[0] != b'\n';
            let mut pos = end;
            let mut fragments: Vec<Vec<u8>> = Vec::new();
            while pos > 0 {
                let count = (256 * 1024u64).min(pos);
                pos -= count;
                file.seek(SeekFrom::Start(pos))?;
                let mut bytes = vec![0; count as usize];
                let read = file.read(&mut bytes)?;
                bytes.truncate(read);
                let parts: Vec<&[u8]> = bytes.split(|b| *b == b'\n').collect();
                fragments.push(parts.last().unwrap().to_vec());
                if parts.len() == 1 {
                    continue;
                }
                let raw: Vec<u8> = fragments
                    .iter()
                    .rev()
                    .flat_map(|v| v.iter().copied())
                    .collect();
                let offset = pos + (bytes.len() - parts.last().unwrap().len()) as u64;
                if skip_partial {
                    skip_partial = false;
                } else if raw.iter().any(|b| !b.is_ascii_whitespace())
                    && !visitor(RawRow {
                        offset: base + offset,
                        bytes: raw,
                    })?
                {
                    return Ok(());
                }
                let mut part_end = bytes.len() - parts.last().unwrap().len() - 1;
                for raw in parts[1..parts.len() - 1].iter().rev() {
                    let part_start = part_end - raw.len();
                    if raw.iter().any(|b| !b.is_ascii_whitespace())
                        && !visitor(RawRow {
                            offset: base + pos + part_start as u64,
                            bytes: raw.to_vec(),
                        })?
                    {
                        return Ok(());
                    }
                    part_end = part_start - 1;
                }
                fragments = vec![parts[0].to_vec()];
            }
            let raw: Vec<u8> = fragments
                .iter()
                .rev()
                .flat_map(|v| v.iter().copied())
                .collect();
            if !skip_partial
                && raw.iter().any(|b| !b.is_ascii_whitespace())
                && !visitor(RawRow {
                    offset: base,
                    bytes: raw,
                })?
            {
                return Ok(());
            }
        }
        Ok(())
    }
    pub fn metadata(&self, every: usize) -> io::Result<(Vec<(Step, usize, u64)>, bool)> {
        if let Some(hints) = &self.hints {
            return Ok(hints.clone());
        }
        let meta_path = self.path.with_extension("meta.json");
        if let Ok(text) = fs::read_to_string(&meta_path) {
            let mut malformed = false;
            if let Some(meta) =
                (!super::super::json::exceeds_integer_limit(text.as_bytes(), self.integer_limit))
                    .then(|| serde_json::from_str::<serde_json::Value>(&text).ok())
                    .flatten()
            {
                malformed = !meta.is_object();
                if meta.get("schema").and_then(|v| v.as_u64()) == Some(2)
                    && meta.get("size").and_then(|v| v.as_u64()) == Some(self.limit)
                {
                    malformed = true;
                    if let (Some(sorted), Some(duplicate), Some(entries)) = (
                        meta["sorted"].as_bool(),
                        meta["has_duplicate_steps"].as_bool(),
                        meta["index"].as_array(),
                    ) {
                        let mut valid = true;
                        let mut anchors = Vec::new();
                        let mut previous: Option<(Step, usize, u64)> = None;
                        for entry in entries {
                            let decoded = entry.as_array().filter(|v| v.len() == 3).and_then(|v| {
                                Some((
                                    Step::parse(&v[0].to_string())?,
                                    v[1].as_u64()? as usize,
                                    v[2].as_u64()?,
                                ))
                            });
                            if let Some(anchor) = decoded {
                                if anchor.2 >= self.limit
                                    || previous.as_ref().is_some_and(|p| {
                                        anchor.1 <= p.1
                                            || anchor.2 <= p.2
                                            || (sorted
                                                && (anchor.0 < p.0
                                                    || (!duplicate && anchor.0 == p.0)))
                                    })
                                {
                                    valid = false;
                                    break;
                                }
                                previous = Some(anchor.clone());
                                anchors.push(anchor);
                            } else {
                                valid = false;
                                break;
                            }
                        }
                        if valid {
                            return Ok((anchors, duplicate || !sorted));
                        }
                    }
                }
            } else {
                self.warn(format!(
                    "Ignoring corrupted metadata in {}",
                    meta_path.display()
                ));
            }
            if malformed {
                self.warn(format!(
                    "Ignoring malformed metadata in {}",
                    meta_path.display()
                ));
            }
        }
        let mut anchors = Vec::new();
        let mut line = 0;
        let mut next = 0;
        let mut previous = None;
        let mut merge = false;
        self.visit(0, true, |row| {
            if let Some(record) = self.parse_row(&row.bytes) {
                let step = record.step().unwrap();
                if previous.as_ref().is_some_and(|p| step <= *p) {
                    merge = true;
                }
                previous = Some(step.clone());
                if line >= next {
                    anchors.push((step, line, row.offset));
                    next = line + every;
                }
            }
            line += 1;
            Ok(())
        })?;
        Ok((anchors, merge))
    }
    pub fn query_encoded(&self, query: &HistoryQuery) -> io::Result<Vec<String>> {
        if query.limit == Some(0) {
            return Ok(Vec::new());
        }
        let (index, needs_merge) = self.metadata(1000)?;
        if query.merge.unwrap_or(needs_merge) {
            return Ok(self
                .clone()
                .with_hints(index, needs_merge)
                .query(query)?
                .into_iter()
                .map(|record| record.encode())
                .collect());
        }
        if query.start.is_none() && query.end.is_none() {
            if let Some(count) = query.limit {
                let mut encoded = Vec::with_capacity(count.min(4096));
                self.reverse(|row| {
                    if self.validated_step(&row.bytes).is_some() {
                        encoded.push(String::from_utf8(row.bytes).expect("validated UTF-8"));
                    }
                    Ok(encoded.len() < count)
                })?;
                encoded.reverse();
                return Ok(encoded);
            }
        }
        let offset = query
            .start
            .as_ref()
            .map(|step| self.offset_step(step))
            .transpose()?
            .unwrap_or(0);
        let mut encoded = VecDeque::new();
        self.visit_until(offset, true, |row| {
            if let Some(step) = self.validated_step(&row.bytes) {
                if query.end.as_ref().is_some_and(|end| step >= *end) {
                    return Ok(false);
                }
                if query.start.as_ref().is_none_or(|start| step >= *start) {
                    encoded.push_back(String::from_utf8(row.bytes).expect("validated UTF-8"));
                    if query.limit.is_some_and(|limit| encoded.len() > limit) {
                        encoded.pop_front();
                    }
                }
            }
            Ok(true)
        })?;
        Ok(encoded.into())
    }
    pub fn query(&self, query: &HistoryQuery) -> io::Result<Vec<JsonObject>> {
        if query.limit == Some(0) {
            return Ok(Vec::new());
        }
        let (_, needs_merge) = self.metadata(1000)?;
        let merge = query.merge.unwrap_or(needs_merge);
        let mut selected: Option<HashSet<Step>> = None;
        if query.start.is_none() && query.end.is_none() {
            if let Some(count) = query.limit {
                if !merge {
                    let mut records = Vec::new();
                    self.reverse(|row| {
                        if let Some(record) = self.parse_row(&row.bytes) {
                            records.push(record);
                        }
                        Ok(records.len() < count)
                    })?;
                    records.reverse();
                    return Ok(records);
                }
                let mut keys = HashSet::new();
                self.reverse(|row| {
                    if let Some(record) = self.parse_row(&row.bytes) {
                        keys.insert(record.step().unwrap());
                    }
                    Ok(keys.len() < count)
                })?;
                selected = Some(keys);
            }
        }
        let mut merged = BTreeMap::<Step, JsonObject>::new();
        let mut records = VecDeque::new();
        let offset = if !merge {
            query
                .start
                .as_ref()
                .map(|s| self.offset_step(s))
                .transpose()?
                .unwrap_or(0)
        } else {
            0
        };
        self.visit_until(offset, true, |row| {
            if let Some(record) = self.parse_row(&row.bytes) {
                let step = record.step().unwrap();
                if !merge && query.end.as_ref().is_some_and(|s| step >= *s) {
                    return Ok(false);
                }
                if query.start.as_ref().is_some_and(|s| step < *s)
                    || query.end.as_ref().is_some_and(|s| step >= *s)
                    || selected.as_ref().is_some_and(|keys| !keys.contains(&step))
                {
                    return Ok(true);
                }
                if merge {
                    merged.entry(step).or_default().merge(&record);
                } else {
                    records.push_back(record);
                    if let Some(count) = query.limit {
                        if records.len() > count {
                            records.pop_front();
                        }
                    }
                }
            }
            Ok(true)
        })?;
        if merge {
            let mut result: Vec<_> = merged.into_values().collect();
            if let Some(limit) = query.limit {
                if result.len() > limit {
                    result.drain(..result.len() - limit);
                }
            }
            Ok(result)
        } else {
            Ok(records.into())
        }
    }
    pub fn offset_step(&self, step: &Step) -> io::Result<u64> {
        let (index, _) = self.metadata(1000)?;
        let start = index.iter().rev().find(|e| e.0 <= *step).map_or(0, |e| e.2);
        let mut found = self.limit;
        self.visit_until(start, true, |row| {
            if let Some(record) = self.parse_row(&row.bytes) {
                if record.step().unwrap() >= *step {
                    found = row.offset;
                    return Ok(false);
                }
            }
            Ok(true)
        })?;
        Ok(found)
    }
    pub fn offset_line(&self, line: usize) -> io::Result<u64> {
        if line == 0 {
            return Ok(0);
        }
        let (index, _) = self.metadata(1000)?;
        let anchor = index.iter().rev().find(|e| e.1 <= line);
        let (mut count, start) = anchor.map_or((0, 0), |e| (e.1, e.2));
        let mut found = self.limit;
        self.visit_until(start, true, |row| {
            if count == line {
                found = row.offset;
                return Ok(false);
            }
            count += 1;
            Ok(true)
        })?;
        Ok(found)
    }
}

/// Native service entry point: no Python, pending bytes, or torn trailing rows.
pub fn visit_rows(path: &Path, mut visitor: impl FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
    SegmentedReader::open(path)?.visit(0, false, |row| visitor(&row.bytes))
}
/// Incremental complete-prefix scan. Blank lines advance the boundary too.
pub fn visit_rows_from(
    path: &Path,
    start: u64,
    mut visitor: impl FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<u64> {
    let reader = SegmentedReader::open(path)?;
    let mut base = 0;
    let mut boundary = start.min(reader.size);
    for (path, length) in &reader.segments {
        if base + length <= start {
            base += length;
            continue;
        }
        let local_start = start.saturating_sub(base);
        let mut file = open_data(path)?;
        file.seek(SeekFrom::Start(local_start))?;
        let mut input = BufReader::new(file.take(length - local_start));
        let mut pos = local_start;
        loop {
            let mut line = Vec::new();
            let n = input.read_until(b'\n', &mut line)?;
            if n == 0 || line.last() != Some(&b'\n') {
                break;
            }
            pos += n as u64;
            boundary = base + pos;
            if line.iter().any(|b| !b.is_ascii_whitespace()) {
                visitor(&line)?;
            }
        }
        if pos < *length {
            break;
        }
        base += length;
    }
    Ok(boundary)
}
pub fn parse_row(bytes: &[u8]) -> Option<JsonObject> {
    let text = std::str::from_utf8(bytes).ok()?;
    let object = JsonObject::parse(text)?;
    object.step()?;
    Some(object)
}
#[derive(Clone, Debug, Default)]
pub struct HistoryQuery {
    pub limit: Option<usize>,
    pub start: Option<Step>,
    pub end: Option<Step>,
    pub merge: Option<bool>,
}
pub fn discover_runs(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut result = Vec::new();
    fn walk(path: &Path, result: &mut Vec<PathBuf>) -> io::Result<()> {
        if !path.is_dir() {
            return Ok(());
        }
        let mut run = false;
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let p = entry.path();
            if p.is_dir()
                && !p
                    .file_name()
                    .is_some_and(|s| s.to_string_lossy().ends_with(".parts"))
            {
                walk(&p, result)?;
            } else if p.file_name().is_some_and(|s| {
                s.to_string_lossy().starts_with("metrics")
                    && s.to_string_lossy().ends_with(".jsonl")
            }) {
                run = true;
            }
        }
        if run {
            result.push(path.to_owned());
        }
        Ok(())
    }
    walk(root, &mut result)?;
    result.sort();
    Ok(result)
}
