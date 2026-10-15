use super::{
    alert_explain, alert_resolve, alert_validate, attr, eval, kind, log, parser, plan, text_in,
    text_out, PyDict,
};
use pyo3::prelude::*;

#[pyfunction]
pub(super) fn alert_rule_metadata(
    node: &Bound<'_, PyAny>,
) -> PyResult<(Vec<Py<PyAny>>, bool, usize)> {
    let mut metrics = vec![];
    let mut time_based = false;
    let mut window = 1;
    let mut stack = vec![node.clone()];
    while let Some(node) = stack.pop() {
        match kind(&node)?.as_str() {
            "MetricRef" => {
                let name = node.getattr("name")?;
                let present = metrics
                    .iter()
                    .any(|m: &Py<PyAny>| m.bind(node.py()).eq(&name).unwrap_or(false));
                if !present {
                    metrics.push(name.unbind());
                }
            }
            "Call" => {
                let name: String = attr(&node, "func")?;
                time_based |= matches!(name.as_str(), "no_data" | "elapsed" | "age");
            }
            "RangeRef" => {
                if let Some(count) = attr::<Option<usize>>(&node, "count")? {
                    window = window.max(count);
                }
            }
            _ => {}
        }
        for child in node.call_method0("children")?.try_iter()? {
            stack.push(child?);
        }
    }
    Ok((metrics, time_based, window))
}
#[pyfunction]
pub(super) fn alert_rules_for_step<'py>(
    rules: &Bound<'py, PyDict>,
    record: Option<&Bound<'py, PyDict>>,
) -> PyResult<Vec<Py<PyAny>>> {
    if rules.is_empty() {
        return Ok(Vec::new());
    }
    let keys: Vec<String> = if let Some(record) = record {
        record
            .keys()
            .iter()
            .map(|key| text_in(&key))
            .collect::<PyResult<Vec<_>>>()?
            .into_iter()
            .filter(|k| !k.starts_with('_'))
            .collect()
    } else {
        vec![]
    };
    let mut selected = vec![];
    for (_, compiled) in rules.iter() {
        let rule = compiled.getattr("rule")?;
        if !attr::<bool>(&rule, "enabled")? {
            continue;
        }
        let time_based: bool = attr(&compiled, "time_based")?;
        if record.is_none() {
            if time_based {
                selected.push(compiled.unbind());
            }
            continue;
        }
        let metrics: Vec<String> = compiled
            .getattr("metrics")?
            .try_iter()?
            .map(|m| text_in(&m?))
            .collect::<PyResult<_>>()?;
        if time_based
            || metrics.is_empty()
            || metrics
                .iter()
                .any(|m| keys.contains(m) || keys.contains(&m.replace('.', "/")))
        {
            selected.push(compiled.unbind());
        }
    }
    Ok(selected)
}
#[pyfunction]
pub(super) fn alert_unresolved(
    compiled: &Bound<'_, PyAny>,
    series: &Bound<'_, PyAny>,
) -> PyResult<Vec<Py<PyAny>>> {
    let mut missing = vec![];
    for metric in compiled.getattr("metrics")?.try_iter()? {
        let name = metric?;
        if alert_resolve(series, &name)?.is_none() {
            missing.push(name.unbind());
        }
    }
    missing.sort_by(|a, b| {
        a.bind(series.py())
            .compare(b.bind(series.py()))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(missing)
}
#[pyfunction]
pub(super) fn alert_evaluate_rule(
    compiled: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
    record: &Bound<'_, PyAny>,
    fire: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let py = compiled.py();
    let rule = compiled.getattr("rule")?;
    let state = compiled.getattr("state")?;
    let expression = compiled.getattr("plan")?;
    let evaluation = if expression.is_none() {
        eval(&compiled.getattr("node")?, ctx)
    } else {
        expression
            .extract::<PyRef<'_, plan::AlertExpression>>()?
            .evaluate(ctx)
    };
    let result = match evaluation {
        Ok(value) => value.truth(),
        Err(err) => {
            let mut state = state.extract::<PyRefMut<'_, AlertRuleState>>()?;
            let disabled = state.failed();
            let name = rule.getattr("name")?.repr()?.to_string();
            log(py, format!("Alert rule {name:?} failed to evaluate: {err}"));
            if disabled {
                rule.setattr("enabled", false)?;
                if let Ok(logger) = py.import("loguru").and_then(|m| m.getattr("logger")) {
                    let _ = logger.call_method1(
                        "error",
                        (format!(
                            "Alert rule {name:?} disabled after {} errors.",
                            state.errors
                        ),),
                    );
                }
            }
            return Ok(());
        }
    };
    let now: f64 = py.import("time")?.call_method0("time")?.extract()?;
    let action = state
        .extract::<PyRefMut<'_, AlertRuleState>>()?
        .transition(result, now, &rule)?;
    if action == 1 {
        if fire.call1((compiled, ctx, record, false))?.is_truthy()? {
            state.extract::<PyRefMut<'_, AlertRuleState>>()?.fired(now);
        }
    } else if action == 2 {
        fire.call1((compiled, ctx, record, true))?;
    }
    Ok(())
}
#[pyfunction]
pub(super) fn alert_template_values<'py>(
    compiled: &Bound<'py, PyAny>,
    ctx: &Bound<'py, PyAny>,
    record: &Bound<'py, PyAny>,
    run_info: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = compiled.py();
    let values = if record.is_none() || !record.is_truthy()? {
        ctx.getattr("record")?.cast::<PyDict>()?.copy()?
    } else {
        record.cast::<PyDict>()?.copy()?
    };
    let rule = compiled.getattr("rule")?;
    values.set_item("name", rule.getattr("name")?)?;
    values.set_item("condition", rule.getattr("condition")?)?;
    let step = ctx.getattr("step")?;
    if step.is_none() {
        values.set_item("step", "-")?;
    } else {
        values.set_item("step", step)?;
    }
    values.set_item(
        "time",
        py.import("time")?
            .call_method1("strftime", ("%Y-%m-%d %H:%M:%S",))?,
    )?;
    values.set_item(
        "project",
        run_info
            .get_item("project")?
            .unwrap_or("".into_pyobject(py)?.into_any()),
    )?;
    values.set_item(
        "run",
        run_info
            .get_item("run")?
            .unwrap_or("".into_pyobject(py)?.into_any()),
    )?;
    let node = compiled.getattr("node")?;
    let explanation = alert_explain(&node, ctx)
        .or_else(|_| Ok::<_, PyErr>(node.call_method0("to_source")?.unbind()))?;
    values.set_item("expr", explanation)?;
    Ok(values)
}
#[pyfunction]
pub(super) fn alert_fire(
    compiled: &Bound<'_, PyAny>,
    ctx: &Bound<'_, PyAny>,
    record: &Bound<'_, PyAny>,
    run_info: &Bound<'_, PyDict>,
    dispatcher: &Bound<'_, PyAny>,
    recovered: bool,
) -> PyResult<bool> {
    let py = compiled.py();
    let rule = compiled.getattr("rule")?;
    let values = alert_template_values(compiled, ctx, record, run_info)?;
    let formatter = py
        .import("expr_tracker.alerts.engine")?
        .getattr("_format")?;
    let rendered = match formatter.call1((rule.getattr("message")?, &values)) {
        Ok(value) => value,
        Err(err) => {
            log(
                py,
                format!(
                    "Failed to render message for rule {}: {err}",
                    rule.getattr("name")?
                ),
            );
            rule.getattr("condition")?
        }
    };
    let title = rule.getattr("title")?;
    let title = if title.is_truthy()? {
        formatter.call1((&title, &values))?
    } else if attr::<bool>(&rule, "auto_named").unwrap_or(false) {
        rule.getattr("condition")?
    } else {
        rule.getattr("name")?
    };
    let kwargs = PyDict::new(py);
    if recovered {
        kwargs.set_item(
            "title",
            text_out(py, &format!("[recovered] {}", text_in(&title)?))?,
        )?;
        kwargs.set_item(
            "text",
            text_out(
                py,
                &format!(
                    "Condition no longer holds: {}",
                    text_in(&values.get_item("expr")?.unwrap())?
                ),
            )?,
        )?;
        kwargs.set_item("level", "info")?;
    } else {
        kwargs.set_item("title", title)?;
        kwargs.set_item("text", rendered)?;
        kwargs.set_item("level", rule.getattr("level")?)?;
    }
    kwargs.set_item(
        "subtitle",
        run_info
            .get_item("run")?
            .unwrap_or(py.None().into_bound(py)),
    )?;
    let fields = PyDict::new(py);
    fields.set_item(
        "project",
        run_info
            .get_item("project")?
            .unwrap_or(py.None().into_bound(py)),
    )?;
    fields.set_item(
        "run",
        run_info
            .get_item("run")?
            .unwrap_or(py.None().into_bound(py)),
    )?;
    fields.set_item("step", values.get_item("step")?.unwrap())?;
    fields.set_item("condition", rule.getattr("condition")?)?;
    kwargs.set_item("fields", fields)?;
    kwargs.set_item("tags", rule.getattr("tags")?.call_method0("copy")?)?;
    kwargs.set_item(
        "link",
        run_info
            .get_item("link")?
            .unwrap_or(py.None().into_bound(py)),
    )?;
    let name = text_in(&rule.getattr("name")?)?;
    kwargs.set_item("source", text_out(py, &format!("rule:{name}"))?)?;
    kwargs.set_item(
        "dedup_key",
        text_out(
            py,
            &format!(
                "rule:{name}:{}:{}",
                if recovered { "recovered" } else { "fire" },
                attr::<usize>(&compiled.getattr("state")?, "fires")?
            ),
        )?,
    )?;
    let message = py
        .import("expr_tracker.alerts.models")?
        .getattr("AlertMessage")?
        .call((), Some(&kwargs))?;
    if let Err(err) = dispatcher.call_method1("send", (message, rule.getattr("channels")?)) {
        log(
            py,
            format!("Failed to dispatch alert for rule {name:?}: {err}"),
        );
        return Ok(false);
    }
    Ok(true)
}

#[pyclass(module = "expr_tracker._native")]
pub(super) struct AlertRuleRegistry {
    rules: Py<PyDict>,
}
#[pymethods]
impl AlertRuleRegistry {
    #[new]
    pub(super) fn new(py: Python<'_>) -> Self {
        Self {
            rules: PyDict::new(py).unbind(),
        }
    }
    #[getter]
    pub(super) fn rules(&self, py: Python<'_>) -> Py<PyDict> {
        self.rules.clone_ref(py)
    }
    pub(super) fn add(
        &self,
        py: Python<'_>,
        rule: &Bound<'_, PyAny>,
        context_factory: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        let node = parser::parse_public(py, &rule.getattr("condition")?)?;
        alert_validate(node.bind(py))?;
        let (metrics, time_based, window) = alert_rule_metadata(node.bind(py))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("rule", rule)?;
        kwargs.set_item("node", node)?;
        kwargs.set_item(
            "plan",
            Py::new(
                py,
                plan::AlertExpression::compile(kwargs.get_item("node")?.as_ref().unwrap())?,
            )?,
        )?;
        kwargs.set_item(
            "metrics",
            py.import("builtins")?.getattr("set")?.call1((metrics,))?,
        )?;
        kwargs.set_item("time_based", time_based)?;
        kwargs.set_item("window", window)?;
        let compiled = py
            .import("expr_tracker.alerts.engine")?
            .getattr("CompiledRule")?
            .call((), Some(&kwargs))?;
        let series = context_factory
            .getattr("series")
            .or_else(|_| context_factory.call1((py.None(),))?.getattr("series"))?;
        if let Ok(ensure) = series.getattr("ensure_capacity") {
            ensure.call1((window,))?;
        }
        let name = rule.getattr("name")?;
        if self.rules.bind(py).contains(&name)? {
            log(
                py,
                format!("Alert rule {name:?} already exists and was replaced."),
            );
        }
        self.rules.bind(py).set_item(name, &compiled)?;
        Ok(compiled.unbind())
    }
    pub(super) fn remove(&self, py: Python<'_>, name: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.rules.bind(py).contains(name)? {
            return Ok(false);
        }
        self.rules.bind(py).del_item(name)?;
        Ok(true)
    }
    pub(super) fn list(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.rules
            .bind(py)
            .iter()
            .map(|(_, compiled)| Ok(compiled.getattr("rule")?.unbind()))
            .collect()
    }
    pub(super) fn time_based(&self, py: Python<'_>) -> PyResult<bool> {
        for (_, compiled) in self.rules.bind(py).iter() {
            if attr::<bool>(&compiled, "time_based")? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    pub(super) fn evaluate(
        &self,
        py: Python<'_>,
        context_factory: &Bound<'_, PyAny>,
        record: &Bound<'_, PyAny>,
        fire: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let record_dict = if record.is_none() {
            None
        } else {
            Some(record.cast::<PyDict>()?)
        };
        let selected = alert_rules_for_step(self.rules.bind(py), record_dict)?;
        if selected.is_empty() {
            return Ok(());
        }
        let ctx = context_factory.call1((record,))?;
        for compiled in selected {
            let compiled = compiled.bind(py);
            let lock = compiled.getattr("lock")?;
            lock.call_method0("__enter__")?;
            let result = alert_evaluate_rule(compiled, &ctx, record, fire);
            lock.call_method1("__exit__", (py.None(), py.None(), py.None()))?;
            result?;
        }
        Ok(())
    }
    pub(super) fn stats(&self, py: Python<'_>, series: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
        let result = PyDict::new(py);
        for (name, compiled) in self.rules.bind(py).iter() {
            let stats = PyDict::new(py);
            let state = compiled.getattr("state")?;
            stats.set_item("fires", state.getattr("fires")?)?;
            stats.set_item("firing", state.getattr("firing")?)?;
            stats.set_item("enabled", compiled.getattr("rule")?.getattr("enabled")?)?;
            stats.set_item(
                "unresolved_metrics",
                if series.is_none() {
                    vec![]
                } else {
                    alert_unresolved(&compiled, series)?
                },
            )?;
            result.set_item(name, stats)?;
        }
        Ok(result.unbind())
    }
    pub(super) fn report_unresolved(
        &self,
        py: Python<'_>,
        series: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        if series.is_none() {
            return Ok(());
        }
        let names = series.call_method0("names")?;
        if !names.is_truthy()? {
            return Ok(());
        }
        for (name, compiled) in self.rules.bind(py).iter() {
            let missing = alert_unresolved(&compiled, series)?;
            if missing.is_empty() {
                continue;
            }
            let mut hints = String::new();
            for metric in &missing {
                let matches: Vec<String> = py
                    .import("difflib")?
                    .getattr("get_close_matches")?
                    .call1((metric, &names, 3, 0.6))?
                    .try_iter()?
                    .map(|m| text_in(&m?))
                    .collect::<PyResult<_>>()?;
                if !matches.is_empty() {
                    hints.push_str(&format!(" Did you mean: {}?", matches.join(", ")));
                }
            }
            let missing = missing
                .into_iter()
                .map(|m| m.bind(py).repr().map(|s| s.to_string()))
                .collect::<PyResult<Vec<_>>>()?
                .join(", ");
            log(
                py,
                format!(
                    "Alert rule {} never fired: metric(s) {missing} were never logged.{hints}",
                    name.repr()?
                ),
            );
        }
        Ok(())
    }
}

#[pyclass(module = "expr_tracker._native")]
#[derive(Default)]
pub(super) struct AlertRuleState {
    #[pyo3(get, set)]
    consecutive: usize,
    #[pyo3(get, set)]
    firing: bool,
    #[pyo3(get, set)]
    last_fire: Option<f64>,
    #[pyo3(get, set)]
    fires: usize,
    #[pyo3(get, set)]
    errors: usize,
}
#[pymethods]
impl AlertRuleState {
    #[new]
    pub(super) fn new() -> Self {
        Self::default()
    }
    /// Returns 1 for a breach, 2 for recovery, and 0 for no notification.
    pub(super) fn transition(
        &mut self,
        result: Option<bool>,
        now: f64,
        rule: &Bound<'_, PyAny>,
    ) -> PyResult<u8> {
        self.errors = 0;
        match result {
            None => Ok(0),
            Some(false) => {
                self.consecutive = 0;
                let recovering = self.firing;
                self.firing = false;
                Ok(if recovering && attr(rule, "notify_recovery")? {
                    2
                } else {
                    0
                })
            }
            Some(true) => {
                self.consecutive += 1;
                if self.consecutive < attr::<usize>(rule, "for_steps")? {
                    return Ok(0);
                }
                let mode: String = attr(rule, "mode")?;
                if mode == "edge" && self.firing {
                    return Ok(0);
                }
                if mode == "level" {
                    if let (Some(last), Some(cooldown)) =
                        (self.last_fire, attr::<Option<f64>>(rule, "cooldown")?)
                    {
                        if cooldown != 0.0 && now - last < cooldown {
                            return Ok(0);
                        }
                    }
                }
                if attr::<Option<usize>>(rule, "max_fires")?.is_some_and(|max| self.fires >= max) {
                    return Ok(0);
                }
                Ok(1)
            }
        }
    }
    pub(super) fn fired(&mut self, now: f64) {
        self.firing = true;
        self.last_fire = Some(now);
        self.fires += 1;
    }
    pub(super) fn failed(&mut self) -> bool {
        self.errors += 1;
        self.errors >= 3
    }
}
