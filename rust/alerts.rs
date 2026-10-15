//! Alert evaluation, transitions and delivery scheduling.
use parking_lot::{Condvar, Mutex};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(FromPyObject)]
enum PointStep {
    Python(Py<PyAny>),
    Decimal(String),
}
impl PointStep {
    fn number(&self, py: Python<'_>) -> f64 {
        match self {
            Self::Python(value) => value.bind(py).extract::<f64>().unwrap_or(f64::NAN),
            Self::Decimal(value) => value.parse().unwrap_or(f64::NAN),
        }
    }
    fn difference(&self, other: &Self, py: Python<'_>) -> Option<f64> {
        match (self, other) {
            (Self::Decimal(a), Self::Decimal(b)) => {
                if let (Ok(a), Ok(b)) = (a.parse::<i128>(), b.parse::<i128>()) {
                    if let Some(span) = a.checked_sub(b) {
                        return Some(span as f64);
                    }
                }
                let integer = py.import("builtins").ok()?.getattr("int").ok()?;
                integer
                    .call1((a,))
                    .ok()?
                    .call_method1("__sub__", (integer.call1((b,)).ok()?,))
                    .ok()?
                    .extract()
                    .ok()
            }
            (Self::Python(a), Self::Python(b)) => a
                .bind(py)
                .call_method1("__sub__", (b.bind(py),))
                .ok()?
                .extract()
                .ok(),
            _ => None,
        }
    }
}
#[derive(FromPyObject)]
struct Point(PointStep, f64, f64);
#[path = "alerts_parser.rs"]
mod parser;
#[path = "alerts_plan.rs"]
mod plan;

// Python permits lone UTF-16 surrogates. Escape them injectively inside Rust,
// and restore them only at the Python string boundary.
const TEXT_ESCAPE: char = '\u{f0000}';
fn text_in(value: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = value.extract::<String>() {
        return Ok(if text.contains(TEXT_ESCAPE) {
            text.replace(TEXT_ESCAPE, &format!("{TEXT_ESCAPE}{TEXT_ESCAPE}"))
        } else {
            text
        });
    }
    let bytes: Vec<u8> = value
        .call_method1("encode", ("utf-8", "surrogatepass"))?
        .extract()?;
    let mut text = String::new();
    let mut rest = bytes.as_slice();
    while !rest.is_empty() {
        match std::str::from_utf8(rest) {
            Ok(valid) => {
                for c in valid.chars() {
                    text.push(c);
                    if c == TEXT_ESCAPE {
                        text.push(c);
                    }
                }
                break;
            }
            Err(err) => {
                let (valid, invalid) = rest.split_at(err.valid_up_to());
                for c in std::str::from_utf8(valid).unwrap().chars() {
                    text.push(c);
                    if c == TEXT_ESCAPE {
                        text.push(c);
                    }
                }
                if invalid.len() < 3
                    || invalid[0] != 0xed
                    || !(0xa0..=0xbf).contains(&invalid[1])
                    || !(0x80..=0xbf).contains(&invalid[2])
                {
                    return Err(PyValueError::new_err("Invalid surrogatepass string"));
                }
                let unit = ((invalid[0] as u32 & 15) << 12)
                    | ((invalid[1] as u32 & 63) << 6)
                    | (invalid[2] as u32 & 63);
                text.push(TEXT_ESCAPE);
                text.push(char::from_u32(0xe000 + unit - 0xd800).unwrap());
                rest = &invalid[3..];
            }
        }
    }
    Ok(text)
}
fn text_out(py: Python<'_>, text: &str) -> PyResult<Py<PyAny>> {
    if !text.contains(TEXT_ESCAPE) {
        return Ok(text.into_pyobject(py)?.into_any().unbind());
    }
    let mut chars = text.chars();
    let mut bytes = vec![];
    while let Some(c) = chars.next() {
        if c == TEXT_ESCAPE {
            if let Some(next) = chars.next() {
                if (0xe000..=0xe7ff).contains(&(next as u32)) {
                    let unit = 0xd800 + (next as u32) - 0xe000;
                    bytes.extend([
                        0xe0 | ((unit >> 12) as u8),
                        0x80 | (((unit >> 6) & 63) as u8),
                        0x80 | ((unit & 63) as u8),
                    ]);
                    continue;
                }

                let mut buf = [0; 4];
                bytes.extend_from_slice(next.encode_utf8(&mut buf).as_bytes());
                continue;
            }
        }
        let mut buf = [0; 4];
        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
    }
    Ok(pyo3::types::PyBytes::new(py, &bytes)
        .call_method1("decode", ("utf-8", "surrogatepass"))?
        .unbind())
}

fn original_position(text: &str, position: usize) -> usize {
    let mut chars = text.chars().take(position);
    let mut count = 0;
    while let Some(c) = chars.next() {
        count += 1;
        if c == TEXT_ESCAPE {
            chars.next();
        }
    }
    count
}

#[derive(Clone, Copy)]
enum Value {
    Unknown,
    Number(f64),
    Bool(bool),
}
impl Value {
    fn number(v: f64) -> Self {
        if v.is_finite() {
            Self::Number(v)
        } else {
            Self::Unknown
        }
    }
    fn num(self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(n),
            Self::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
    fn truth(self) -> Option<bool> {
        self.num().filter(|n| n.is_finite()).map(|n| n != 0.0)
    }
    fn python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            Self::Unknown => py
                .import("expr_tracker.alerts.expr.functions")?
                .getattr("UNKNOWN")?
                .unbind(),
            Self::Number(n) => n.into_pyobject(py)?.into_any().unbind(),
            Self::Bool(b) => b.into_pyobject(py)?.to_owned().into_any().unbind(),
        })
    }
}

fn kind(node: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(node.get_type().name()?.to_string())
}
fn attr<T: for<'a, 'py> FromPyObject<'a, 'py>>(o: &Bound<'_, PyAny>, key: &str) -> PyResult<T> {
    o.getattr(key)?.extract().map_err(Into::into)
}
fn error(py: Python<'_>, text: impl Into<String>) -> PyErr {
    let text = text.into();
    match py
        .import("expr_tracker.alerts.expr.eval")
        .and_then(|m| m.getattr("ExprError"))
    {
        Ok(cls) => PyErr::from_value(cls.call1((text,)).unwrap()),
        Err(_) => PyValueError::new_err(text),
    }
}
fn resolved(ctx: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let value = ctx.call_method1("resolve", (text_out(ctx.py(), name)?,))?;
    if value.is_none() {
        Ok(None)
    } else {
        Ok(Some(text_in(&value)?))
    }
}
fn metric_name(node: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    match kind(node)?.as_str() {
        "MetricRef" => Ok(Some(text_in(&node.getattr("name")?)?)),
        "RangeRef" => Ok(Some(text_in(&node.getattr("ref")?.getattr("name")?)?)),
        _ => Ok(None),
    }
}
fn points(
    node: &Bound<'_, PyAny>,
    context: &plan::Context<'_, '_>,
    fallback: Option<usize>,
) -> PyResult<Vec<Point>> {
    let Some(name) = metric_name(node)? else {
        return Err(error(node.py(), "Expected a metric"));
    };
    let (count, duration) = if kind(node)? == "RangeRef" {
        (attr(node, "count")?, attr(node, "duration")?)
    } else {
        (fallback.map(|n| n as i64), None)
    };
    context.points_name(&name, count, duration)
}

fn function_meta(name: &str) -> Option<(&'static str, usize, Option<usize>, Option<usize>)> {
    Some(match name {
        "mean" | "std" | "var" | "median" | "sum" | "first" | "count" | "slope" | "zscore"
        | "increasing" | "decreasing" => ("window", 1, Some(1), None),
        "last" => ("window", 1, Some(1), Some(1)),
        "diff" | "rate" | "pct_change" => ("window", 1, Some(1), Some(2)),
        "ema" | "stalled" => ("window", 1, Some(2), None),
        "min" | "max" => ("dual", 1, None, None),
        "abs" | "exp" | "sqrt" | "floor" | "ceil" => ("scalar", 1, Some(1), None),
        "log" => ("scalar", 1, Some(2), None),
        "step" | "elapsed" => ("special", 0, Some(0), None),
        "no_data" | "age" | "has" | "isnan" | "isinf" => ("special", 1, Some(1), None),
        _ => return None,
    })
}
fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    // Partial sums preserve small contributions alongside much larger ones.
    let mut partials: Vec<f64> = vec![];
    for mut x in values {
        if !x.is_finite() {
            return x;
        }
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    partials.iter().rev().sum()
}

mod delivery;
mod evaluation;
mod rules;
mod watchdog;
use delivery::*;
use evaluation::*;
use rules::*;
use watchdog::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    parser::register(module)?;
    plan::register(module)?;
    module.add_function(wrap_pyfunction!(alert_evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(alert_validate, module)?)?;
    module.add_function(wrap_pyfunction!(alert_explain, module)?)?;
    module.add_function(wrap_pyfunction!(alert_window, module)?)?;
    module.add_function(wrap_pyfunction!(alert_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(alert_function_meta, module)?)?;
    module.add_function(wrap_pyfunction!(alert_tristate, module)?)?;
    module.add_function(wrap_pyfunction!(alert_resolve, module)?)?;
    module.add_function(wrap_pyfunction!(alert_accepts, module)?)?;
    module.add_function(wrap_pyfunction!(alert_select_channels, module)?)?;
    module.add_function(wrap_pyfunction!(alert_rule_metadata, module)?)?;
    module.add_function(wrap_pyfunction!(alert_rules_for_step, module)?)?;
    module.add_function(wrap_pyfunction!(alert_unresolved, module)?)?;
    module.add_function(wrap_pyfunction!(alert_evaluate_rule, module)?)?;
    module.add_function(wrap_pyfunction!(alert_template_values, module)?)?;
    module.add_function(wrap_pyfunction!(alert_fire, module)?)?;
    module.add_function(wrap_pyfunction!(alert_flush_channels, module)?)?;
    module.add_class::<AlertRuleState>()?;
    module.add_class::<AlertTimer>()?;
    module.add_class::<AlertTokenBucket>()?;
    module.add_class::<AlertDeduper>()?;
    module.add_class::<AlertChannel>()?;
    module.add_class::<AlertRuleRegistry>()?;
    Ok(())
}
