use super::{
    attr, compensated_sum, error, function_meta, kind, metric_name, plan, points, text_in,
    text_out, Point, PyList, Value,
};
use pyo3::prelude::*;

pub(super) fn window_value(py: Python<'_>, name: &str, points: &[Point], args: &[f64]) -> Value {
    let n = points.len();
    if name == "count" {
        return Value::Number(n as f64);
    }
    let need = match name {
        "std" | "var" | "diff" | "rate" | "pct_change" | "slope" | "zscore" | "stalled"
        | "increasing" | "decreasing" => 2,
        _ => 1,
    };
    if n < need {
        return Value::Unknown;
    }
    let vals: Vec<f64> = points.iter().map(|p| p.2).collect();
    let first = vals[0];
    let last = vals[n - 1];
    let mean = || compensated_sum(vals.iter().copied()) / n as f64;
    let variance = || {
        let m = mean();
        compensated_sum(vals.iter().map(|v| (v - m).powi(2))) / n as f64
    };
    let value = match name {
        "mean" => mean(),
        "std" => variance().sqrt(),
        "var" => variance(),
        "median" => {
            let mut v = vals.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if n % 2 == 1 {
                v[n / 2]
            } else {
                (v[n / 2 - 1] + v[n / 2]) / 2.0
            }
        }
        "sum" => vals.iter().sum(),
        "min" => vals
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
            .unwrap(),
        "max" => vals
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
            .unwrap(),
        "first" => first,
        "last" => last,
        "diff" => last - first,
        "rate" => {
            let span = points[n - 1].0.difference(&points[0].0, py);
            let Some(span) = span else {
                return Value::Unknown;
            };
            if span <= 0.0 {
                return Value::Unknown;
            }
            (last - first) / span
        }
        "pct_change" => {
            if first == 0.0 {
                return Value::Unknown;
            }
            (last - first) / first.abs()
        }
        "slope" => {
            let xs: Vec<f64> = points.iter().map(|p| p.0.number(py)).collect();
            let mx = compensated_sum(xs.iter().copied()) / n as f64;
            let my = mean();
            let denom: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
            if denom == 0.0 {
                return Value::Unknown;
            }
            xs.iter()
                .zip(points)
                .map(|(x, p)| (x - mx) * (p.2 - my))
                .sum::<f64>()
                / denom
        }
        "zscore" => {
            let std = variance().sqrt();
            if std == 0.0 {
                return Value::Unknown;
            }
            (last - mean()) / std
        }
        "ema" => {
            let alpha = args.first().copied().unwrap_or(0.3);
            if !(alpha > 0.0 && alpha <= 1.0) {
                return Value::Unknown;
            }
            vals[1..]
                .iter()
                .fold(first, |v, p| alpha * p + (1.0 - alpha) * v)
        }
        "stalled" => {
            if vals.iter().any(|v| v.is_nan()) {
                return Value::Unknown;
            }
            let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
            return Value::Bool(max - min <= args.first().copied().unwrap_or(0.0));
        }
        "increasing" | "decreasing" => {
            if vals.iter().any(|v| !v.is_finite()) {
                return Value::Unknown;
            }
            return Value::Bool(vals.windows(2).all(|p| {
                if name == "increasing" {
                    p[1] > p[0]
                } else {
                    p[1] < p[0]
                }
            }));
        }
        _ => return Value::Unknown,
    };
    Value::number(value)
}
pub(super) fn scalar(name: &str, args: &[f64]) -> Value {
    if args.is_empty() {
        return Value::Unknown;
    }
    let a = args[0];
    Value::number(match name {
        "abs" => a.abs(),
        "log" => {
            if args.len() == 2 {
                a.ln() / args[1].ln()
            } else {
                a.ln()
            }
        }
        "exp" => a.exp(),
        "sqrt" => a.sqrt(),
        "floor" => a.floor(),
        "ceil" => a.ceil(),
        "min" => args
            .iter()
            .copied()
            .reduce(|a, b| if b < a { b } else { a })
            .unwrap(),
        "max" => args
            .iter()
            .copied()
            .reduce(|a, b| if b > a { b } else { a })
            .unwrap(),
        _ => f64::NAN,
    })
}
pub(super) fn eval(node: &Bound<'_, PyAny>, ctx: &Bound<'_, PyAny>) -> PyResult<Value> {
    eval_with(node, &plan::Context::generic(ctx))
}
pub(super) fn eval_with(
    node: &Bound<'_, PyAny>,
    context: &plan::Context<'_, '_>,
) -> PyResult<Value> {
    let node_kind = kind(node)?;
    let result = eval_inner(node, context, &node_kind);
    if node_kind == "Call" {
        if let Err(err) = &result {
            let cls = node
                .py()
                .import("expr_tracker.alerts.expr.eval")?
                .getattr("ExprError")?;
            if !err.value(node.py()).is_instance(&cls)? {
                return Ok(Value::Unknown);
            }
        }
    }
    result
}
pub(super) fn eval_inner(
    node: &Bound<'_, PyAny>,
    context: &plan::Context<'_, '_>,
    node_kind: &str,
) -> PyResult<Value> {
    let ctx = context.python();
    let py = node.py();
    Ok(match node_kind {
        "Literal" => {
            let v = node.getattr("value")?;
            if v.is_instance_of::<pyo3::types::PyBool>() {
                Value::Bool(v.extract()?)
            } else {
                Value::number(v.extract()?)
            }
        }
        "MetricRef" => {
            let name = text_in(&node.getattr("name")?)?;
            context
                .latest_name(&name)?
                .map_or(Value::Unknown, |p| Value::number(p.2))
        }
        "RangeRef" => points(node, context, None)?
            .last()
            .map_or(Value::Unknown, |p| Value::Number(p.2)),
        "Not" => eval_with(&node.getattr("operand")?, context)?
            .truth()
            .map_or(Value::Unknown, |b| Value::Bool(!b)),
        "UnaryOp" => eval_with(&node.getattr("operand")?, context)?
            .num()
            .map_or(Value::Unknown, |n| Value::Number(-n)),
        "BoolOp" => {
            let op: String = attr(node, "op")?;
            let mut unknown = false;
            let mut decisive = false;
            for v in node.getattr("values")?.try_iter()? {
                match eval_with(&v?, context)?.truth() {
                    None => unknown = true,
                    Some(b) => {
                        if b == (op == "or") {
                            decisive = true;
                        }
                    }
                }
            }
            if decisive {
                Value::Bool(op == "or")
            } else if unknown {
                Value::Unknown
            } else {
                Value::Bool(op != "or")
            }
        }
        "Compare" | "BinOp" => {
            let a = eval_with(&node.getattr("left")?, context)?.num();
            let b = eval_with(&node.getattr("right")?, context)?.num();
            let op: String = attr(node, "op")?;
            match (a, b) {
                (Some(a), Some(b)) if a.is_finite() && b.is_finite() => match op.as_str() {
                    ">" => Value::Bool(a > b),
                    ">=" => Value::Bool(a >= b),
                    "<" => Value::Bool(a < b),
                    "<=" => Value::Bool(a <= b),
                    "==" => Value::Bool(a == b),
                    "!=" => Value::Bool(a != b),
                    "+" => Value::number(a + b),
                    "-" => Value::number(a - b),
                    "*" => Value::number(a * b),
                    "/" => Value::number(a / b),
                    "%" => {
                        if b == 0.0 {
                            Value::Unknown
                        } else {
                            let mut rem = a % b;
                            if rem != 0.0 && rem.is_sign_negative() != b.is_sign_negative() {
                                rem += b;
                            }
                            if rem == 0.0 {
                                rem = 0.0f64.copysign(b);
                            }
                            Value::number(rem)
                        }
                    }
                    _ => return Err(error(py, format!("Unknown operator {op:?}"))),
                },
                _ => Value::Unknown,
            }
        }
        "Call" => {
            let name: String = attr(node, "func")?;
            let args = node.getattr("args")?;
            let args = args.cast::<PyList>()?;
            let Some((kind, _, _, fallback)) = function_meta(&name) else {
                return Err(error(py, format!("Unknown function {name:?}")));
            };
            if kind == "special" {
                match name.as_str() {
                    "step" => {
                        let step: Option<f64> = attr(ctx, "step")?;
                        step.map_or(Value::Unknown, Value::Number)
                    }
                    "elapsed" => Value::Number(
                        (attr::<f64>(ctx, "now")? - attr::<f64>(ctx, "started_at")?).max(0.0),
                    ),
                    "no_data" => {
                        let threshold = eval_with(&args.get_item(0)?, context)?.num();
                        let reference = attr::<Option<f64>>(ctx, "last_commit_time")?
                            .unwrap_or(attr(ctx, "started_at")?);
                        threshold.map_or(Value::Unknown, |t| {
                            Value::Bool(
                                attr::<f64>(ctx, "now").unwrap_or(reference) - reference >= t,
                            )
                        })
                    }
                    _ => {
                        let metric = args.get_item(0)?;
                        let Some(m) = metric_name(&metric)? else {
                            return Err(error(py, format!("{name}() expects a metric name")));
                        };
                        if name == "has" {
                            Value::Bool(context.has_name(&m)?)
                        } else if let Some(p) = context.latest_name(&m)? {
                            match name.as_str() {
                                "age" => Value::Number((attr::<f64>(ctx, "now")? - p.1).max(0.0)),
                                "isnan" => Value::Bool(p.2.is_nan()),
                                "isinf" => Value::Bool(p.2.is_infinite()),
                                _ => Value::Unknown,
                            }
                        } else {
                            Value::Unknown
                        }
                    }
                }
            } else {
                let window = kind == "window"
                    || (kind == "dual"
                        && args.len() == 1
                        && metric_name(&args.get_item(0)?)?.is_some());
                let mut extra = vec![];
                let mut unknown = false;
                for arg in args.iter().skip(if window { 1 } else { 0 }) {
                    match eval_with(&arg, context)?.num() {
                        Some(v) => extra.push(v),
                        None => unknown = true,
                    }
                }
                if unknown {
                    Value::Unknown
                } else if window {
                    window_value(
                        py,
                        &name,
                        &points(&args.get_item(0)?, context, fallback)?,
                        &extra,
                    )
                } else {
                    scalar(&name, &extra)
                }
            }
        }
        _ => return Err(error(py, format!("Cannot evaluate node {}", node.repr()?))),
    })
}

#[pyfunction]
pub(super) fn alert_evaluate(
    py: Python<'_>,
    node: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    eval(node, ctx)?.python(py)
}

#[pyfunction]
pub(super) fn alert_window(
    py: Python<'_>,
    name: &str,
    points: Vec<Point>,
    args: Vec<f64>,
) -> PyResult<Py<PyAny>> {
    window_value(py, name, &points, &args).python(py)
}

#[pyfunction]
pub(super) fn alert_scalar(py: Python<'_>, name: &str, args: Vec<f64>) -> PyResult<Py<PyAny>> {
    scalar(name, &args).python(py)
}

#[pyfunction]
pub(super) fn alert_function_meta(
    name: &str,
) -> PyResult<(String, usize, Option<usize>, Option<usize>)> {
    function_meta(name)
        .map(|(kind, low, high, window)| (kind.into(), low, high, window))
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(name.to_string()))
}
#[pyfunction]
pub(super) fn alert_tristate(value: &Bound<'_, PyAny>) -> PyResult<Option<bool>> {
    if kind(value)? == "_Unknown" {
        return Ok(None);
    }
    Ok(Value::number(value.extract()?).truth())
}
#[pyfunction]
pub(super) fn alert_resolve(
    series: &Bound<'_, PyAny>,
    name: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<PyAny>>> {
    if series.call_method1("has", (name,))?.is_truthy()? {
        return Ok(Some(name.clone().unbind()));
    }
    let alternative = name.call_method1("replace", (".", "/"))?;
    if !alternative.eq(name)? && series.call_method1("has", (&alternative,))?.is_truthy()? {
        return Ok(Some(alternative.unbind()));
    }
    Ok(None)
}
#[pyfunction]
pub(super) fn alert_validate(node: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = node.py();
    for child in node.call_method0("children")?.try_iter()? {
        alert_validate(&child?)?;
    }
    if kind(node)? != "Call" {
        return Ok(());
    }
    let name: String = attr(node, "func")?;
    let args = node.getattr("args")?;
    let args = args.cast::<PyList>()?;
    let Some((kind, low, high, fallback)) = function_meta(&name) else {
        let names = py
            .import("expr_tracker.alerts.expr.functions")?
            .getattr("FUNCTIONS")?;
        let matches: Vec<String> = py
            .import("difflib")?
            .getattr("get_close_matches")?
            .call1((&name, names, 3, 0.6))?
            .extract()?;
        let hint = if matches.is_empty() {
            String::new()
        } else {
            format!(" Did you mean: {}?", matches.join(", "))
        };
        return Err(error(py, format!("Unknown function {name:?}.{hint}")));
    };
    if args.len() < low || high.is_some_and(|h| args.len() > h) {
        let expected = if high == Some(low) {
            low.to_string()
        } else {
            format!("{low}..{}", high.map_or("N".into(), |h| h.to_string()))
        };
        return Err(error(
            py,
            format!("{name}() takes {expected} argument(s), got {}", args.len()),
        ));
    }
    if kind == "window" {
        let first = args.get_item(0)?;
        if metric_name(&first)?.is_none() {
            return Err(error(
                py,
                format!(
                    "{name}() expects a metric as its first argument, got {}",
                    first.call_method0("to_source")?
                ),
            ));
        }
        if self::kind(&first)? == "MetricRef" && fallback.is_none() {
            return Err(error(
                py,
                format!(
                    "{name}() requires an explicit window, e.g. {name}({}[20])",
                    first.call_method0("to_source")?
                ),
            ));
        }
    }
    if kind == "dual" && metric_name(&args.get_item(0)?)?.is_none() && args.len() < 2 {
        return Err(error(
            py,
            format!(
                "{name}() needs a window (e.g. {name}(loss[20])) or at least two scalar arguments"
            ),
        ));
    }
    if kind == "special"
        && matches!(name.as_str(), "has" | "age" | "isnan" | "isinf")
        && metric_name(&args.get_item(0)?)?.is_none()
    {
        return Err(error(py, format!("{name}() expects a metric name")));
    }
    Ok(())
}

pub(super) fn annotate(node: &Bound<'_, PyAny>, ctx: &Bound<'_, PyAny>) -> PyResult<String> {
    let source = text_in(&node.call_method0("to_source")?)?;
    if kind(node)? == "Literal" {
        return Ok(source);
    }
    let value = eval(node, ctx)?;
    let value = match value {
        Value::Unknown => "?".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => node
            .py()
            .import("builtins")?
            .getattr("format")?
            .call1((n, ".6g"))?
            .extract()?,
    };
    Ok(format!("{source}={value}"))
}
pub(super) fn explain_text(node: &Bound<'_, PyAny>, ctx: &Bound<'_, PyAny>) -> PyResult<String> {
    match kind(node)?.as_str() {
        "BoolOp" => {
            let op: String = attr(node, "op")?;
            let values: PyResult<Vec<_>> = node
                .getattr("values")?
                .try_iter()?
                .map(|v| explain_text(&v?, ctx))
                .collect();
            Ok(values?.join(&format!(" {op} ")))
        }
        "Not" => Ok(format!(
            "not {}",
            explain_text(&node.getattr("operand")?, ctx)?
        )),
        "Compare" => Ok(format!(
            "{} {} {}",
            annotate(&node.getattr("left")?, ctx)?,
            attr::<String>(node, "op")?,
            annotate(&node.getattr("right")?, ctx)?
        )),
        _ => annotate(node, ctx),
    }
}

#[pyfunction]
pub(super) fn alert_explain(
    node: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    text_out(node.py(), &explain_text(node, ctx)?)
}
