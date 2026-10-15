//! Compiled expression plans keep Python AST reflection off the committed-step path.
use super::{
    attr, error, function_meta, kind, resolved, scalar, text_in, text_out, window_value, Point,
    PointStep, Value,
};
use crate::history::NativeMetricSeries;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::Write;

fn json_key(name: &str) -> String {
    fn character(output: &mut String, value: u32) {
        match value {
            0x22 => output.push_str("\\\""),
            0x5c => output.push_str("\\\\"),
            0x08 => output.push_str("\\b"),
            0x0c => output.push_str("\\f"),
            0x0a => output.push_str("\\n"),
            0x0d => output.push_str("\\r"),
            0x09 => output.push_str("\\t"),
            0x20..=0x7e => output.push(char::from_u32(value).unwrap()),
            0..=0xffff => {
                write!(output, "\\u{value:04x}").unwrap();
            }
            _ => {
                let value = value - 0x10000;
                write!(
                    output,
                    "\\u{:04x}\\u{:04x}",
                    0xd800 + (value >> 10),
                    0xdc00 + (value & 1023)
                )
                .unwrap();
            }
        }
    }
    let mut output = String::with_capacity(name.len() + 2);
    output.push('"');
    let mut chars = name.chars();
    while let Some(mut ch) = chars.next() {
        if ch == super::TEXT_ESCAPE {
            if let Some(next) = chars.next() {
                if (0xe000..=0xe7ff).contains(&(next as u32)) {
                    character(&mut output, 0xd800 + next as u32 - 0xe000);
                    continue;
                }
                ch = next;
            }
        }
        character(&mut output, ch as u32);
    }
    output.push('"');
    output
}

struct Metric {
    name: String,
    key: String,
    alias: String,
}
impl Metric {
    fn compile(name: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self::from_name(text_in(name)?))
    }
    fn from_name(name: String) -> Self {
        let key = json_key(&name);
        let alias = if name.contains('.') {
            json_key(&name.replace('.', "/"))
        } else {
            key.clone()
        };
        Self { name, key, alias }
    }
}
pub(super) struct Context<'a, 'py> {
    python: &'a Bound<'py, PyAny>,
    native: Option<Py<NativeMetricSeries>>,
}
impl<'a, 'py> Context<'a, 'py> {
    pub(super) fn new(python: &'a Bound<'py, PyAny>) -> Self {
        let native = (|| -> PyResult<Py<NativeMetricSeries>> {
            let resolver = python.getattr("resolve")?.getattr("__func__")?;
            let original = python.getattr("_native_resolve")?.getattr("__func__")?;
            if !resolver.is(&original) {
                return Err(pyo3::exceptions::PyTypeError::new_err("Custom resolver"));
            }
            Ok(python.getattr("series")?.getattr("_native")?.extract()?)
        })()
        .ok();
        Self { python, native }
    }
    pub(super) fn generic(python: &'a Bound<'py, PyAny>) -> Self {
        let mut context = Self::new(python);
        if context.native.is_some() {
            let standard = (|| -> PyResult<bool> {
                let series = python.getattr("series")?;
                let methods = python.getattr("_native_series_methods")?;
                for (index, name) in ["has", "latest", "window_points"].into_iter().enumerate() {
                    if !series
                        .getattr(name)?
                        .getattr("__func__")?
                        .is(&methods.get_item(index)?)
                    {
                        return Ok(false);
                    }
                }
                Ok(true)
            })()
            .unwrap_or(false);
            if !standard {
                context.native = None;
            }
        }
        context
    }
    pub(super) fn python(&self) -> &'a Bound<'py, PyAny> {
        self.python
    }
    pub(super) fn latest_name(&self, name: &str) -> PyResult<Option<Point>> {
        self.latest(&Metric::from_name(name.into()))
    }
    pub(super) fn has_name(&self, name: &str) -> PyResult<bool> {
        Ok(self.resolve(&Metric::from_name(name.into()))?.is_some())
    }
    pub(super) fn points_name(
        &self,
        name: &str,
        count: Option<i64>,
        duration: Option<f64>,
    ) -> PyResult<Vec<Point>> {
        self.points(&Metric::from_name(name.into()), count, duration)
    }
    fn resolve(&self, metric: &Metric) -> PyResult<Option<String>> {
        if let Some(native) = &self.native {
            let series = native.borrow(self.python.py());
            Ok(if series.has_json_key(&metric.key) {
                Some(metric.key.clone())
            } else if series.has_json_key(&metric.alias) {
                Some(metric.alias.clone())
            } else {
                None
            })
        } else {
            resolved(self.python, &metric.name)
        }
    }
    fn latest(&self, metric: &Metric) -> PyResult<Option<Point>> {
        let Some(key) = self.resolve(metric)? else {
            return Ok(None);
        };
        if let Some(native) = &self.native {
            Ok(native
                .borrow(self.python.py())
                .latest_json_key(&key)
                .map(|p| Point(PointStep::Decimal(p.0), p.1, p.2)))
        } else {
            self.python
                .getattr("series")?
                .call_method1("latest", (text_out(self.python.py(), &key)?,))?
                .extract()
        }
    }
    fn points(
        &self,
        metric: &Metric,
        count: Option<i64>,
        duration: Option<f64>,
    ) -> PyResult<Vec<Point>> {
        let Some(key) = self.resolve(metric)? else {
            return Ok(vec![]);
        };
        if let Some(native) = &self.native {
            let points = native
                .borrow(self.python.py())
                .window_json_key(&key, count, duration, Some(attr(self.python, "now")?))
                .unwrap_or_default();
            Ok(points
                .into_iter()
                .map(|p| Point(PointStep::Decimal(p.0), p.1, p.2))
                .collect())
        } else {
            let kwargs = PyDict::new(self.python.py());
            kwargs.set_item("count", count)?;
            kwargs.set_item("duration", duration)?;
            kwargs.set_item("now", self.python.getattr("now")?)?;
            self.python
                .getattr("series")?
                .call_method(
                    "window_points",
                    (text_out(self.python.py(), &key)?,),
                    Some(&kwargs),
                )?
                .extract()
        }
    }
}

enum Plan {
    Literal(Value),
    Metric(Metric),
    Range(Metric, Option<usize>, Option<f64>),
    Call(String, Vec<Plan>),
    Not(Box<Plan>),
    Unary(Box<Plan>),
    Binary(String, Box<Plan>, Box<Plan>),
    Boolean(String, Vec<Plan>),
}
impl Plan {
    fn compile(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(match kind(node)?.as_str() {
            "Literal" => {
                let value = node.getattr("value")?;
                Self::Literal(if value.is_instance_of::<pyo3::types::PyBool>() {
                    Value::Bool(value.extract()?)
                } else {
                    Value::number(value.extract()?)
                })
            }
            "MetricRef" => Self::Metric(Metric::compile(&node.getattr("name")?)?),
            "RangeRef" => Self::Range(
                Metric::compile(&node.getattr("ref")?.getattr("name")?)?,
                attr(node, "count")?,
                attr(node, "duration")?,
            ),
            "Call" => Self::Call(
                attr(node, "func")?,
                node.getattr("args")?
                    .try_iter()?
                    .map(|n| Self::compile(&n?))
                    .collect::<PyResult<_>>()?,
            ),
            "Not" => Self::Not(Box::new(Self::compile(&node.getattr("operand")?)?)),
            "UnaryOp" => Self::Unary(Box::new(Self::compile(&node.getattr("operand")?)?)),
            "BoolOp" => Self::Boolean(
                attr(node, "op")?,
                node.getattr("values")?
                    .try_iter()?
                    .map(|n| Self::compile(&n?))
                    .collect::<PyResult<_>>()?,
            ),
            "BinOp" | "Compare" => Self::Binary(
                attr(node, "op")?,
                Box::new(Self::compile(&node.getattr("left")?)?),
                Box::new(Self::compile(&node.getattr("right")?)?),
            ),
            _ => {
                return Err(error(
                    node.py(),
                    format!("Cannot evaluate node {}", node.repr()?),
                ))
            }
        })
    }
    fn metric(&self) -> Option<&Metric> {
        match self {
            Self::Metric(m) | Self::Range(m, _, _) => Some(m),
            _ => None,
        }
    }
    fn points(&self, ctx: &Context<'_, '_>, fallback: Option<usize>) -> PyResult<Vec<Point>> {
        let Some(metric) = self.metric() else {
            return Err(error(ctx.python.py(), "Expected a metric"));
        };
        let (count, duration) = match self {
            Self::Range(_, count, duration) => (*count, *duration),
            _ => (fallback, None),
        };
        ctx.points(
            metric,
            count.map(|n| i64::try_from(n).unwrap_or(i64::MAX)),
            duration,
        )
    }
    fn evaluate(&self, ctx: &Context<'_, '_>) -> PyResult<Value> {
        let result = self.inner(ctx);
        if matches!(self, Self::Call(..)) {
            if let Err(err) = &result {
                let cls = ctx
                    .python
                    .py()
                    .import("expr_tracker.alerts.expr.eval")?
                    .getattr("ExprError")?;
                if !err.value(ctx.python.py()).is_instance(&cls)? {
                    return Ok(Value::Unknown);
                }
            }
        }
        result
    }
    fn inner(&self, ctx: &Context<'_, '_>) -> PyResult<Value> {
        let py = ctx.python.py();
        let python = ctx.python;
        Ok(match self {
            Self::Literal(value) => *value,
            Self::Metric(metric) => ctx
                .latest(metric)?
                .map_or(Value::Unknown, |p| Value::number(p.2)),
            Self::Range(..) => self
                .points(ctx, None)?
                .last()
                .map_or(Value::Unknown, |p| Value::Number(p.2)),
            Self::Not(node) => node
                .evaluate(ctx)?
                .truth()
                .map_or(Value::Unknown, |b| Value::Bool(!b)),
            Self::Unary(node) => node
                .evaluate(ctx)?
                .num()
                .map_or(Value::Unknown, |n| Value::Number(-n)),
            Self::Boolean(op, nodes) => {
                let mut unknown = false;
                let mut decisive = false;
                for node in nodes {
                    match node.evaluate(ctx)?.truth() {
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
            Self::Binary(op, left, right) => {
                match (left.evaluate(ctx)?.num(), right.evaluate(ctx)?.num()) {
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
            Self::Call(name, args) => {
                let Some((kind, _, _, fallback)) = function_meta(name) else {
                    return Err(error(py, format!("Unknown function {name:?}")));
                };
                if kind == "special" {
                    match name.as_str() {
                        "step" => attr::<Option<f64>>(python, "step")?
                            .map_or(Value::Unknown, Value::Number),
                        "elapsed" => Value::Number(
                            (attr::<f64>(python, "now")? - attr::<f64>(python, "started_at")?)
                                .max(0.0),
                        ),
                        "no_data" => {
                            let threshold = args[0].evaluate(ctx)?.num();
                            let reference = attr::<Option<f64>>(python, "last_commit_time")?
                                .unwrap_or(attr(python, "started_at")?);
                            threshold.map_or(Value::Unknown, |t| {
                                Value::Bool(
                                    attr::<f64>(python, "now").unwrap_or(reference) - reference
                                        >= t,
                                )
                            })
                        }
                        _ => {
                            let Some(metric) = args[0].metric() else {
                                return Err(error(py, format!("{name}() expects a metric name")));
                            };
                            if name == "has" {
                                Value::Bool(ctx.resolve(metric)?.is_some())
                            } else if let Some(point) = ctx.latest(metric)? {
                                match name.as_str() {
                                    "age" => Value::Number(
                                        (attr::<f64>(python, "now")? - point.1).max(0.0),
                                    ),
                                    "isnan" => Value::Bool(point.2.is_nan()),
                                    "isinf" => Value::Bool(point.2.is_infinite()),
                                    _ => Value::Unknown,
                                }
                            } else {
                                Value::Unknown
                            }
                        }
                    }
                } else {
                    let window = kind == "window"
                        || (kind == "dual" && args.len() == 1 && args[0].metric().is_some());
                    let mut extra = vec![];
                    let mut unknown = false;
                    for arg in args.iter().skip(if window { 1 } else { 0 }) {
                        match arg.evaluate(ctx)?.num() {
                            Some(v) => extra.push(v),
                            None => unknown = true,
                        }
                    }
                    if unknown {
                        Value::Unknown
                    } else if window {
                        window_value(py, name, &args[0].points(ctx, fallback)?, &extra)
                    } else {
                        scalar(name, &extra)
                    }
                }
            }
        })
    }
}
#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertExpression {
    root: Plan,
}
#[pymethods]
impl AlertExpression {
    #[new]
    fn new(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        super::alert_validate(node)?;
        Ok(Self {
            root: Plan::compile(node)?,
        })
    }
    #[pyo3(name = "evaluate")]
    fn evaluate_python(&self, ctx: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.evaluate(ctx)?.python(ctx.py())
    }
}
impl AlertExpression {
    pub(super) fn compile(node: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::new(node)
    }
    pub(super) fn evaluate(&self, ctx: &Bound<'_, PyAny>) -> PyResult<Value> {
        self.root.evaluate(&Context::new(ctx))
    }
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<AlertExpression>()?;
    Ok(())
}
