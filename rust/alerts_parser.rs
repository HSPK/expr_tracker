use pyo3::prelude::*;
use pyo3::types::PyList;

#[derive(Clone)]
struct Token {
    kind: &'static str,
    text: String,
    number: Option<f64>,
    pos: usize,
}
fn syntax(py: Python<'_>, message: String, source: &str, pos: usize) -> PyErr {
    let pos = super::original_position(source, pos);
    match py
        .import("expr_tracker.alerts.expr.lexer")
        .and_then(|m| m.getattr("ExprSyntaxError"))
        .and_then(|cls| cls.call1((message.clone(), super::text_out(py, source)?, pos)))
    {
        Ok(value) => PyErr::from_value(value),
        Err(_) => pyo3::exceptions::PyValueError::new_err(message),
    }
}
fn name_char(c: char) -> bool {
    c.is_alphanumeric() || "_.@".contains(c)
}
fn scan(source: &str, py: Python<'_>) -> PyResult<Vec<Token>> {
    let chars: Vec<_> = source.chars().collect();
    let mut i = 0;
    let mut tokens = vec![];
    while i < chars.len() {
        let ch = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        let pos = i;
        if "\"'`".contains(ch) {
            i += 1;
            let start = i;
            while i < chars.len() && chars[i] != ch {
                i += 1;
            }
            if i == chars.len() {
                return Err(syntax(
                    py,
                    format!("Unterminated {ch}quoted name{ch}"),
                    source,
                    pos,
                ));
            }
            tokens.push(Token {
                kind: "NAME",
                text: chars[start..i].iter().collect(),
                number: None,
                pos,
            });
            i += 1;
            continue;
        }
        if ch.is_ascii_digit()
            || (ch == '.' && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit()))
        {
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            if chars.get(i).is_some_and(|c| "eE".contains(*c)) {
                let mut j = i + 1;
                if chars.get(j).is_some_and(|c| "+-".contains(*c)) {
                    j += 1;
                }
                if chars.get(j).is_some_and(|c| c.is_ascii_digit()) {
                    i = j;
                    while chars.get(i).is_some_and(|c| c.is_ascii_digit()) {
                        i += 1;
                    }
                }
            }
            let text: String = chars[pos..i].iter().collect();
            let value = py
                .import("builtins")?
                .getattr("float")?
                .call1((&text,))?
                .extract::<f64>()?;
            let mut duration = None;
            for (unit, scale) in [
                ("min", 60.0),
                ("ms", 0.001),
                ("s", 1.0),
                ("m", 60.0),
                ("h", 3600.0),
                ("d", 86400.0),
            ] {
                let end = i + unit.len();
                if end <= chars.len()
                    && chars[i..end].iter().collect::<String>() == unit
                    && !chars
                        .get(end)
                        .is_some_and(|c| c.is_alphanumeric() || "_.".contains(*c))
                {
                    duration = Some(value * scale);
                    i = end;
                    break;
                }
            }
            tokens.push(Token {
                kind: if duration.is_some() {
                    "DURATION"
                } else {
                    "NUMBER"
                },
                text,
                number: Some(duration.unwrap_or(value)),
                pos,
            });
            continue;
        }
        if ch.is_alphabetic() || ch == '_' {
            while i < chars.len()
                && (name_char(chars[i])
                    || (chars[i] == '/' && chars.get(i + 1).is_some_and(|c| name_char(*c))))
            {
                i += 1;
            }
            let text: String = chars[pos..i].iter().collect();
            tokens.push(Token {
                kind: if ["and", "or", "not"].contains(&text.as_str()) {
                    "OP"
                } else {
                    "NAME"
                },
                text,
                number: None,
                pos,
            });
            continue;
        }
        let pair: String = chars[i..(i + 2).min(chars.len())].iter().collect();
        if [">=", "<=", "==", "!=", "&&", "||"].contains(&pair.as_str()) {
            tokens.push(Token {
                kind: "OP",
                text: pair,
                number: None,
                pos,
            });
            i += 2;
            continue;
        }
        if "><+-*/%()[],|&!~".contains(ch) {
            tokens.push(Token {
                kind: "OP",
                text: ch.to_string(),
                number: None,
                pos,
            });
            i += 1;
            continue;
        }
        return Err(syntax(
            py,
            format!(
                "Unexpected character {}",
                py.import("builtins")?
                    .getattr("repr")?
                    .call1((ch.to_string(),))?
            ),
            source,
            pos,
        ));
    }
    tokens.push(Token {
        kind: "EOF",
        text: "None".into(),
        number: None,
        pos: chars.len(),
    });
    Ok(tokens)
}

struct Parser<'py> {
    py: Python<'py>,
    source: String,
    tokens: Vec<Token>,
    pos: usize,
}
impl<'py> Parser<'py> {
    fn current(&self) -> &Token {
        &self.tokens[self.pos]
    }
    fn matches(&self, ops: &[&str]) -> bool {
        self.current().kind == "OP" && ops.contains(&self.current().text.as_str())
    }
    fn advance(&mut self) -> Token {
        let t = self.current().clone();
        self.pos += 1;
        t
    }
    fn err(&self, message: String) -> PyErr {
        syntax(self.py, message, &self.source, self.current().pos)
    }
    fn expect(&mut self, op: &str) -> PyResult<()> {
        if !self.matches(&[op]) {
            return Err(self.err(format!(
                "Expected {op:?} but found {:?}",
                self.current().text
            )));
        }
        self.advance();
        Ok(())
    }
    fn node<A>(&self, name: &str, args: A) -> PyResult<Py<PyAny>>
    where
        A: pyo3::call::PyCallArgs<'py>,
    {
        self.py
            .import("expr_tracker.alerts.expr.nodes")?
            .getattr(name)?
            .call1(args)
            .map(Bound::unbind)
    }
    fn parse(&mut self) -> PyResult<Py<PyAny>> {
        if self.current().kind == "EOF" {
            return Err(self.err("Empty expression".into()));
        }
        let node = self.or()?;
        if self.current().kind != "EOF" {
            return Err(self.err(format!("Unexpected token {:?}", self.current().text)));
        }
        Ok(node)
    }
    fn or(&mut self) -> PyResult<Py<PyAny>> {
        let mut nodes = vec![self.and()?];
        while self.matches(&["or", "||", "|"]) {
            self.advance();
            nodes.push(self.and()?);
        }
        if nodes.len() == 1 {
            Ok(nodes.pop().unwrap())
        } else {
            let list = PyList::new(self.py, nodes)?;
            self.node("BoolOp", ("or", list))
        }
    }
    fn and(&mut self) -> PyResult<Py<PyAny>> {
        let mut nodes = vec![self.not()?];
        while self.matches(&["and", "&&", "&"]) {
            self.advance();
            nodes.push(self.not()?);
        }
        if nodes.len() == 1 {
            Ok(nodes.pop().unwrap())
        } else {
            let list = PyList::new(self.py, nodes)?;
            self.node("BoolOp", ("and", list))
        }
    }
    fn not(&mut self) -> PyResult<Py<PyAny>> {
        if self.matches(&["not", "!", "~"]) {
            self.advance();
            let node = self.not()?;
            self.node("Not", (node,))
        } else {
            self.compare()
        }
    }
    fn compare(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.arith()?;
        let mut nodes = vec![];
        while self.matches(&[">", ">=", "<", "<=", "==", "!="]) {
            let op = self.advance().text;
            let right = self.arith()?;
            nodes.push(self.node("Compare", (op, left, right.clone_ref(self.py)))?);
            left = right;
        }
        if nodes.is_empty() {
            Ok(left)
        } else if nodes.len() == 1 {
            Ok(nodes.pop().unwrap())
        } else {
            let list = PyList::new(self.py, nodes)?;
            self.node("BoolOp", ("and", list))
        }
    }
    fn arith(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.term()?;
        while self.matches(&["+", "-"]) {
            let op = self.advance().text;
            let right = self.term()?;
            left = self.node("BinOp", (op, left, right))?;
        }
        Ok(left)
    }
    fn term(&mut self) -> PyResult<Py<PyAny>> {
        let mut left = self.unary()?;
        while self.matches(&["*", "/", "%"]) {
            let op = self.advance().text;
            let right = self.unary()?;
            left = self.node("BinOp", (op, left, right))?;
        }
        Ok(left)
    }
    fn unary(&mut self) -> PyResult<Py<PyAny>> {
        if self.matches(&["+", "-"]) {
            let op = self.advance().text;
            let node = self.unary()?;
            if op == "+" {
                Ok(node)
            } else {
                self.node("UnaryOp", ("-", node))
            }
        } else {
            self.postfix()
        }
    }
    fn postfix(&mut self) -> PyResult<Py<PyAny>> {
        let mut node = self.primary()?;
        if self.matches(&["["]) {
            if super::kind(node.bind(self.py))? != "MetricRef" {
                return Err(self.err("Window selector [...] can only follow a metric name".into()));
            }
            self.advance();
            let token = self.advance();
            let value = token.number;
            let (count, duration) = match (token.kind, value) {
                ("DURATION", Some(value)) => (None, Some(value)),
                ("NUMBER", Some(value)) => {
                    if value <= 0.0 || value.fract() != 0.0 || !value.is_finite() {
                        return Err(syntax(
                            self.py,
                            "Window size must be a positive integer or a duration like 30s".into(),
                            &self.source,
                            token.pos,
                        ));
                    }
                    (Some(value as usize), None)
                }
                _ => {
                    self.pos -= 1;
                    return Err(self.err("Expected a window size such as [20] or [30s]".into()));
                }
            };
            self.expect("]")?;
            node = self.node("RangeRef", (node, count, duration))?;
        }
        Ok(node)
    }
    fn primary(&mut self) -> PyResult<Py<PyAny>> {
        if self.matches(&["("]) {
            self.advance();
            let node = self.or()?;
            self.expect(")")?;
            return Ok(node);
        }
        let token = self.current().clone();
        if ["NUMBER", "DURATION"].contains(&token.kind) {
            self.advance();
            return self.node("Literal", (token.number.unwrap(),));
        }
        if token.kind == "NAME" {
            self.advance();
            if token.text == "true" || token.text == "false" {
                return self.node("Literal", (if token.text == "true" { 1.0 } else { 0.0 },));
            }
            if self.matches(&["("]) {
                self.advance();
                let mut args = vec![];
                if !self.matches(&[")"]) {
                    loop {
                        args.push(self.or()?);
                        if self.matches(&[","]) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.expect(")")?;
                let list = PyList::new(self.py, args)?;
                return self.node("Call", (token.text, list));
            }
            return self.node("MetricRef", (super::text_out(self.py, &token.text)?,));
        }
        Err(self.err(format!("Unexpected token {:?}", token.text)))
    }
}
#[pyfunction]
fn alert_parse(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let source = super::text_in(source)?;
    Parser {
        py,
        tokens: scan(&source, py)?,
        source,
        pos: 0,
    }
    .parse()
}
pub(super) fn parse_public(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    alert_parse(py, source)
}
#[pyfunction]
fn alert_tokenize(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let source = super::text_in(source)?;
    let cls = py
        .import("expr_tracker.alerts.expr.lexer")?
        .getattr("Token")?;
    let tokens = PyList::empty(py);
    for token in scan(&source, py)? {
        let value = if let Some(number) = token.number {
            number.into_pyobject(py)?.into_any().unbind()
        } else if token.kind == "EOF" {
            py.None()
        } else {
            super::text_out(py, &token.text)?
        };
        tokens.append(cls.call1((
            token.kind,
            value,
            super::original_position(&source, token.pos),
        ))?)?;
    }
    Ok(tokens.into_any().unbind())
}
fn top_level(text: &str, separator: &str, split: bool) -> (i64, Vec<String>) {
    let mut depth = 0;
    let mut quote = None;
    let mut parts = vec![];
    let mut start = 0;
    for (pos, ch) in text.char_indices() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
            }
        } else if "\"'`".contains(ch) {
            quote = Some(ch);
        } else if "([".contains(ch) {
            depth += 1;
        } else if ")]".contains(ch) {
            depth -= 1;
        } else if depth == 0 && text[pos..].starts_with(separator) {
            if !split {
                return (text[..pos].chars().count() as i64, vec![]);
            }
            parts.push(text[start..pos].into());
            start = pos + separator.len();
        }
    }
    parts.push(text[start..].into());
    (-1, parts)
}
fn is_level(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "debug" | "info" | "warn" | "warning" | "err" | "error" | "crit" | "critical" | "fatal"
    )
}
fn split_rule_text(text: &str) -> PyResult<(String, Option<String>, String)> {
    let source = text.trim();
    if source.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("Empty alert rule"));
    }
    let arrow = top_level(source, "=>", false).0;
    if arrow >= 0 {
        let byte_pos = source.char_indices().nth(arrow as usize).unwrap().0;
        let condition = source[..byte_pos].trim().to_string();
        let rest = source[byte_pos + 2..].trim();
        if let Some((head, tail)) = rest.split_once(':') {
            let head = head.trim();
            if head.is_empty() {
                return Ok((condition, None, tail.trim().into()));
            }
            if is_level(head) {
                return Ok((condition, Some(head.into()), tail.trim().into()));
            }
            return Ok((condition, None, rest.into()));
        }
        return Ok(if is_level(rest) {
            (condition, Some(rest.into()), String::new())
        } else {
            (condition, None, rest.into())
        });
    }
    let (_, parts) = top_level(source, ",", true);
    let condition = parts[0].trim().to_string();
    if condition.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Alert rule has no condition: {text:?}"
        )));
    }
    if parts.len() < 2 {
        return Ok((condition, None, String::new()));
    }
    let candidate = parts[1].trim();
    Ok(if is_level(candidate) {
        (
            condition,
            Some(candidate.into()),
            parts[2..].join(",").trim().into(),
        )
    } else {
        (condition, None, parts[1..].join(",").trim().into())
    })
}
#[pyfunction]
fn alert_scan_name(source: &Bound<'_, PyAny>, mut i: usize) -> PyResult<usize> {
    let source = super::text_in(source)?;
    let target = i;
    i = 0;
    let mut position = 0;
    let mut scan = source.chars();
    while position < target {
        let Some(ch) = scan.next() else {
            break;
        };
        i += 1;
        position += 1;
        if ch == super::TEXT_ESCAPE {
            scan.next();
            i += 1;
        }
    }
    let chars: Vec<_> = source.chars().collect();
    while i < chars.len()
        && (name_char(chars[i])
            || (chars[i] == '/' && chars.get(i + 1).is_some_and(|c| name_char(*c))))
    {
        i += 1;
    }
    Ok(super::original_position(&source, i))
}
#[pyfunction]
fn alert_top_level(
    py: Python<'_>,
    text: &Bound<'_, PyAny>,
    separator: &str,
    split: bool,
) -> PyResult<(i64, Vec<Py<PyAny>>)> {
    let text = super::text_in(text)?;
    let (index, parts) = top_level(&text, separator, split);
    Ok((
        if index < 0 {
            index
        } else {
            super::original_position(&text, index as usize) as i64
        },
        parts
            .iter()
            .map(|s| super::text_out(py, s))
            .collect::<PyResult<_>>()?,
    ))
}
#[pyfunction]
fn alert_split_rule(
    py: Python<'_>,
    text: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyAny>, Option<Py<PyAny>>, Py<PyAny>)> {
    let (condition, level, message) = split_rule_text(&super::text_in(text)?)?;
    Ok((
        super::text_out(py, &condition)?,
        level.map(|s| super::text_out(py, &s)).transpose()?,
        super::text_out(py, &message)?,
    ))
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(alert_parse, module)?)?;
    module.add_function(wrap_pyfunction!(alert_tokenize, module)?)?;
    module.add_function(wrap_pyfunction!(alert_top_level, module)?)?;
    module.add_function(wrap_pyfunction!(alert_split_rule, module)?)?;
    module.add_function(wrap_pyfunction!(alert_scan_name, module)?)?;
    Ok(())
}
