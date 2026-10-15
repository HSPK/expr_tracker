use indexmap::IndexMap;
use std::cmp::Ordering;

/// Arbitrary precision integer step. Its ordering is numeric, not lexical.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Step(pub String);
impl Step {
    pub fn parse(text: &str) -> Option<Self> {
        let text = text.trim();
        if text == "true" {
            return Some(Self("1".into()));
        }
        if text == "false" {
            return Some(Self("0".into()));
        }
        let digits = text.strip_prefix('-').unwrap_or(text);
        if digits.is_empty()
            || !digits.bytes().all(|b| b.is_ascii_digit())
            || (digits.len() > 1 && digits.starts_with('0'))
        {
            return None;
        }
        Some(Self(if text == "-0" {
            "0".into()
        } else {
            text.into()
        }))
    }
    pub fn next(&self) -> Self {
        Self(add_int(&self.0, "1"))
    }
}
impl Ord for Step {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.0.strip_prefix('-');
        let b = other.0.strip_prefix('-');
        match (a, b) {
            (Some(a), Some(b)) => a.len().cmp(&b.len()).then(a.cmp(b)).reverse(),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => self.0.len().cmp(&other.0.len()).then(self.0.cmp(&other.0)),
        }
    }
}
impl PartialOrd for Step {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Ordered object whose values retain their original JSON spelling. Python JSON
/// permits nonfinite numbers and unpaired UTF-16 escapes; neither is coerced.
#[derive(Clone, Debug, Default)]
pub struct JsonObject(pub IndexMap<String, String>);
impl JsonObject {
    pub fn parse(text: &str) -> Option<Self> {
        let mut result = Self::default();
        visit_object(text, |key, value| {
            result.set(decode_key(key)?, value.to_owned());
            Some(())
        })?;
        Some(result)
    }
    pub fn validated_step(text: &str) -> Option<Step> {
        let mut step = None;
        visit_object(text, |key, value| {
            if key == "\"_step\"" || key.contains('\\') && decode_key(key)?.as_str() == "_step" {
                step = Step::parse(value);
            }
            Some(())
        })?;
        step
    }
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }
    pub fn set(&mut self, key: String, value: String) {
        self.0.insert(key, value);
    }
    pub fn remove(&mut self, key: &str) {
        self.0.shift_remove(key);
    }
    pub fn merge(&mut self, other: &Self) {
        for (key, value) in &other.0 {
            self.set(key.clone(), value.clone());
        }
    }
    pub fn step(&self) -> Option<Step> {
        Step::parse(self.get("_step")?)
    }
    pub fn encode(&self) -> String {
        let capacity = self
            .0
            .iter()
            .map(|(k, v)| k.len() + v.len() + 6)
            .sum::<usize>();
        let mut text = String::with_capacity(capacity + 2);
        text.push('{');
        for (index, (key, value)) in self.0.iter().enumerate() {
            if index != 0 {
                text.push_str(", ");
            }
            if let Some(raw) = key.strip_prefix("\0S") {
                text.push_str(raw);
            } else if let Some(original) = key.strip_prefix("\0N") {
                text.push_str(&serde_json::to_string(original).unwrap());
            } else {
                text.push_str(&serde_json::to_string(key).unwrap());
            }
            text.push_str(": ");
            text.push_str(value);
        }
        text.push('}');
        text
    }
    /// Browser-compatible view. Lossless SDK reads use `encode` instead.
    pub fn browser_value(&self) -> serde_json::Value {
        parse_compatible_json(self.encode().as_bytes())
            .map(|parsed| encode_browser_value(parsed.value))
            .unwrap_or(serde_json::Value::Null)
    }
}
fn visit_object(text: &str, mut visit: impl FnMut(&str, &str) -> Option<()>) -> Option<()> {
    let mut p = Parser {
        text: text.as_bytes(),
        pos: 0,
    };
    p.space();
    if p.take() != Some(b'{') {
        return None;
    }
    p.space();
    if p.peek() == Some(b'}') {
        p.pos += 1;
    } else {
        loop {
            p.space();
            let start = p.pos;
            p.string()?;
            let key_raw = &text[start..p.pos];
            p.space();
            if p.take() != Some(b':') {
                return None;
            }
            p.space();
            let start = p.pos;
            p.value(0)?;
            visit(key_raw, &text[start..p.pos])?;
            p.space();
            match p.take()? {
                b'}' => break,
                b',' => (),
                _ => return None,
            }
        }
    }
    p.space();
    (p.pos == text.len()).then_some(())
}

fn decode_key(raw: &str) -> Option<String> {
    // A leading NUL distinguishes escaped unpaired surrogates without ever
    // replacing their spelling in the lossless output.
    match serde_json::from_str::<String>(raw) {
        Ok(key) if key.starts_with('\0') => Some(format!("\0N{key}")),
        Ok(key) => Some(key),
        Err(_) => Some(format!("\0S{}", canonical_string(raw)?)),
    }
}

fn canonical_string(raw: &str) -> Option<String> {
    let mut units = Vec::new();
    let mut chars = raw[1..raw.len() - 1].chars();
    while let Some(character) = chars.next() {
        if character == '\\' {
            let escaped = chars.next()?;
            if escaped == 'u' {
                let text: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&text, 16).ok()?);
            } else {
                let character = match escaped {
                    '"' => '"',
                    '\\' => '\\',
                    '/' => '/',
                    'b' => '\u{8}',
                    'f' => '\u{c}',
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    _ => return None,
                };
                units.push(character as u16);
            }
        } else {
            let mut encoded = [0; 2];
            units.extend_from_slice(character.encode_utf16(&mut encoded));
        }
    }
    let mut result = String::from("\"");
    for unit in units {
        match unit {
            34 => result.push_str("\\\""),
            92 => result.push_str("\\\\"),
            32..=126 => result.push(char::from_u32(unit as u32)?),
            _ => result.push_str(&format!("\\u{unit:04x}")),
        }
    }
    result.push('"');
    Some(result)
}

struct Parser<'a> {
    text: &'a [u8],
    pos: usize,
}
impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.get(self.pos).copied()
    }
    fn take(&mut self) -> Option<u8> {
        let value = self.peek();
        self.pos += usize::from(value.is_some());
        value
    }
    fn space(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\r' | b'\n' | b'\t')) {
            self.pos += 1;
        }
    }
    fn string(&mut self) -> Option<()> {
        if self.take()? != b'"' {
            return None;
        }
        loop {
            let remaining = &self.text[self.pos..];
            let ordinary = memchr::memchr2(b'"', b'\\', remaining)?;
            if remaining[..ordinary].iter().any(|byte| *byte < 32) {
                return None;
            }
            self.pos += ordinary;
            match self.take()? {
                b'"' => return Some(()),
                b'\\' => match self.take()? {
                    b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => (),
                    b'u' => {
                        for _ in 0..4 {
                            if !self.take()?.is_ascii_hexdigit() {
                                return None;
                            }
                        }
                    }
                    _ => return None,
                },
                0..=31 => return None,
                _ => (),
            }
        }
    }
    fn value(&mut self, depth: usize) -> Option<()> {
        if depth > 512 {
            return None;
        }
        self.space();
        match self.peek()? {
            b'"' => self.string(),
            b'{' | b'[' => {
                let object = self.take()? == b'{';
                let close = if object { b'}' } else { b']' };
                self.space();
                if self.peek() == Some(close) {
                    self.pos += 1;
                    return Some(());
                }
                loop {
                    if object {
                        self.space();
                        self.string()?;
                        self.space();
                        if self.take()? != b':' {
                            return None;
                        }
                    }
                    self.value(depth + 1)?;
                    self.space();
                    match self.take()? {
                        c if c == close => return Some(()),
                        b',' => (),
                        _ => return None,
                    }
                }
            }
            _ => {
                let start = self.pos;
                while matches!(self.peek(), Some(c) if !matches!(c,b','|b'}'|b']'|b' '|b'\r'|b'\n'|b'\t'))
                {
                    self.pos += 1;
                }
                let token = std::str::from_utf8(&self.text[start..self.pos]).ok()?;
                if matches!(
                    token,
                    "null" | "true" | "false" | "NaN" | "Infinity" | "-Infinity"
                ) {
                    return Some(());
                }
                // serde's arbitrary_precision parser accepts huge finite integers.
                serde_json::from_str::<serde_json::Number>(token)
                    .ok()
                    .map(|_| ())
            }
        }
    }
}

/// Parsed values retain finite arbitrary-precision `Number`s for chart/index
/// computations. Only nonfinite values are tagged at this stage.
pub struct CompatibleJson {
    pub value: serde_json::Value,
    pub escaped_surrogates: bool,
}

pub fn parse_compatible_json(raw: &[u8]) -> Result<CompatibleJson, serde_json::Error> {
    let mut output = Vec::with_capacity(raw.len());
    let mut quoted = false;
    let mut escaped_surrogates = false;
    let mut at = 0;
    while at < raw.len() {
        if quoted {
            if raw[at] == b'\\' {
                if let Some(code) = unicode_escape(raw, at) {
                    let paired = (0xd800..=0xdbff).contains(&code)
                        && unicode_escape(raw, at + 6)
                            .is_some_and(|next| (0xdc00..=0xdfff).contains(&next));
                    if paired {
                        output.extend_from_slice(&raw[at..at + 12]);
                        at += 12;
                        continue;
                    }
                    if (0xd800..=0xdfff).contains(&code) {
                        output.push(b'\\');
                        output.extend_from_slice(&raw[at..at + 6]);
                        escaped_surrogates = true;
                        at += 6;
                        continue;
                    }
                }
                let end = (at + 2).min(raw.len());
                output.extend_from_slice(&raw[at..end]);
                at = end;
                continue;
            }
            if raw[at] == b'"' {
                quoted = false;
            }
        } else if raw[at] == b'"' {
            quoted = true;
        } else {
            let special = [b"-Infinity".as_slice(), b"Infinity", b"NaN"]
                .into_iter()
                .find(|token| {
                    raw[at..].starts_with(token)
                        && raw
                            .get(at + token.len())
                            .is_none_or(|next| next.is_ascii_whitespace() || b",]}".contains(next))
                });
            if let Some(token) = special {
                output.extend_from_slice(br#"{"$et.nonfinite":""#);
                output.extend_from_slice(token);
                output.extend_from_slice(br#""}"#);
                at += token.len();
                continue;
            }
        }
        output.push(raw[at]);
        at += 1;
    }
    Ok(CompatibleJson {
        value: serde_json::from_slice(&output)?,
        escaped_surrogates,
    })
}

fn unicode_escape(raw: &[u8], at: usize) -> Option<u16> {
    let bytes = raw.get(at..at + 6)?;
    if &bytes[..2] != b"\\u" {
        return None;
    }
    u16::from_str_radix(std::str::from_utf8(&bytes[2..]).ok()?, 16).ok()
}

/// Apply JavaScript-safe tags only at the HTTP/serialization boundary.
pub fn encode_browser_value(value: serde_json::Value) -> serde_json::Value {
    use serde_json::{json, Value};
    match value {
        Value::Number(ref number) => {
            let raw = number.to_string();
            if !raw.contains(['.', 'e', 'E']) {
                if raw
                    .parse::<i64>()
                    .is_ok_and(|integer| integer.unsigned_abs() <= 9_007_199_254_740_991)
                {
                    value
                } else {
                    json!({"$et.integer": raw})
                }
            } else if number.as_f64().is_some_and(f64::is_finite) {
                value
            } else {
                json!({"$et.number": raw})
            }
        }
        Value::Array(values) => {
            Value::Array(values.into_iter().map(encode_browser_value).collect())
        }
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, encode_browser_value(value)))
                .collect(),
        ),
        value => value,
    }
}

pub fn number(text: &str) -> Option<f64> {
    match text {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        "true" | "false" | "null" => None,
        _ => text.parse().ok(),
    }
}
pub(super) fn exceeds_integer_limit(bytes: &[u8], limit: usize) -> bool {
    if limit == 0 {
        return false;
    }
    let mut at = 0;
    let mut quoted = false;
    while at < bytes.len() {
        if quoted {
            let Some(next) = memchr::memchr2(b'"', b'\\', &bytes[at..]) else {
                break;
            };
            at += next;
            if bytes[at] == b'\\' {
                at += 2;
                continue;
            }
            if bytes[at] == b'"' {
                quoted = false;
            }
            at += 1;
            continue;
        }
        if bytes[at] == b'"' {
            quoted = true;
            at += 1;
            continue;
        }
        if bytes[at] == b'-' || bytes[at].is_ascii_digit() {
            let start = at;
            at += 1;
            while at < bytes.len() && !b",]} \n\r\t".contains(&bytes[at]) {
                at += 1;
            }
            let token = &bytes[start..at];
            let digits = token.len() - usize::from(token.first() == Some(&b'-'));
            if digits > limit && !token.iter().any(|b| matches!(b, b'.' | b'e' | b'E')) {
                return true;
            }
        } else {
            at += 1;
        }
    }
    false
}
pub fn float_json(value: f64) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value == f64::INFINITY {
        "Infinity".into()
    } else if value == f64::NEG_INFINITY {
        "-Infinity".into()
    } else {
        serde_json::to_string(&value).unwrap()
    }
}

pub fn add_int(a: &str, b: &str) -> String {
    let an = a.starts_with('-');
    let bn = b.starts_with('-');
    let a = a.trim_start_matches('-');
    let b = b.trim_start_matches('-');
    let cmp = a.len().cmp(&b.len()).then(a.cmp(b));
    let (big, small, negative) = if cmp == Ordering::Less {
        (b, a, bn)
    } else {
        (a, b, an)
    };
    let mut out = Vec::new();
    let mut carry = 0i16;
    let mut small = small.bytes().rev();
    for digit in big.bytes().rev() {
        let right = small.next().map_or(0, |v| (v - b'0') as i16);
        let sum = (digit - b'0') as i16 + carry + if an == bn { right } else { -right };
        if sum < 0 {
            out.push((sum + 10) as u8 + b'0');
            carry = -1;
        } else {
            out.push((sum % 10) as u8 + b'0');
            carry = sum / 10;
        }
    }
    if carry > 0 {
        out.push(carry as u8 + b'0');
    }
    while out.len() > 1 && out.last() == Some(&b'0') {
        out.pop();
    }
    if negative && out != [b'0'] {
        out.push(b'-');
    }
    out.reverse();
    String::from_utf8(out).unwrap()
}
