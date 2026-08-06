# Alert expressions

A small language for conditions over metric history. It is parsed by a hand-written
lexer and Pratt parser — not Python's `ast` — so its precedence can be the one that
reads correctly for alerting.

## Grammar

```
rule       := expression "=>" [level] [":" message]
expression := or_expr
or_expr    := and_expr ("or" | "||" | "|" and_expr)*
and_expr   := not_expr ("and" | "&&" | "&" not_expr)*
not_expr   := ["not" | "!"] comparison
comparison := arithmetic (("<" | "<=" | ">" | ">=" | "==" | "!=") arithmetic)?
arithmetic := term (("+" | "-") term)*
term       := factor (("*" | "/" | "%") factor)*
factor     := number | metric | metric "[" window "]" | call | "(" expression ")"
window     := integer | duration        # 20 | 30s | 5m | 2h | 1d
```

Precedence is `or < and < not < comparison < +- < */%`.

!!! warning "Different from Python"
    Python parses `a > 50 | b > 5` as `a > (50 | b) > 5`. Here `|` is a plain `or`,
    so it means `(a > 50) or (b > 5)` — what it looks like.

    Run `et rules explain "<expr>"` if in doubt; it prints the fully parenthesised
    form.

## Metric names

Bare names may contain letters, digits, `_`, `.`, `@` and `/`, so the names ML code
actually uses need no ceremony:

```
train/loss > 1
val/m1/acc@16 < 0.5
mean(train/loss[20]) > 1
```

`.` also resolves to `/`, so `eval.acc` finds `eval/acc` if no metric is literally
called `eval.acc`.

### `/` is part of a name unless it is spaced

A `/` joins the name when a name character follows it immediately. Division must
therefore be written with spaces:

| Written | Means |
| --- | --- |
| `a/b` | the metric `a/b` |
| `a / b` | `a` divided by `b` |
| `a /b`, `a/ b` | `a` divided by `b` (any space makes it division) |
| `loss/2` | the metric `loss/2` |
| `loss / 2` | `loss` divided by 2 |
| `1/2` | division — a name cannot start with a digit |

### Quote anything else

A name with spaces or other characters must be quoted, with `"`, `'` or `` ` ``:

```
"train loss" > 1
'val acc@16' < 0.5
`a-b` > 1
```

Quotes only ever produce a metric name; the language has no string literals.
`to_source()` renders with `"` (or another quote if the name contains one).

## Windows

`metric[window]` selects the recent history of a metric.

| Form | Meaning |
| --- | --- |
| `loss[20]` | the last 20 points |
| `loss[30s]` | the last 30 seconds |
| `loss[5m]`, `loss[2h]`, `loss[1d]` | minutes, hours, days |

Aggregates require an explicit window, because the answer would otherwise depend
silently on buffer size:

```
mean(loss)       # error: mean() requires an explicit window
mean(loss[20])   # fine
diff(loss)       # fine - diff has a sensible default of 2 points
```

## Functions

| Category | Functions |
| --- | --- |
| Aggregate | `mean` `std` `var` `median` `sum` `min` `max` `first` `last` `count` |
| Change | `diff(m[w])` `rate(m[w])` `pct_change(m[w])` |
| Trend | `slope(m[w])` `zscore(m[w])` `ema(m[w], alpha)` `increasing(m[w])` `decreasing(m[w])` |
| Predicate | `isnan(m)` `isinf(m)` `has(m)` `stalled(m[w], eps)` |
| Context | `step()` `elapsed()` `age(m)` `no_data(duration)` |
| Scalar | `abs` `log` `exp` `sqrt` `floor` `ceil` `min` `max` |

`min` and `max` are dual: `min(loss[20])` is a rolling minimum, `min(a, b)` is the
smaller of two scalars. The argument types decide.

`no_data(30s)` takes a **duration** and asks whether anything has been committed
recently. `age(loss)` takes a **metric** and returns how long since it last appeared.

## Three-valued logic

Any expression that cannot be answered evaluates to `UNKNOWN` — never an exception:

- a metric that was never logged
- fewer points than the window needs
- `NaN` or infinity in the input
- division by zero, `log(0)`, `sqrt(-1)`

`UNKNOWN` never fires a rule and never changes its state, so warm-up and sparse
metrics cannot produce false alarms. It propagates by Kleene rules:

| Expression | Result |
| --- | --- |
| `UNKNOWN or true` | `true` |
| `UNKNOWN or false` | `UNKNOWN` |
| `UNKNOWN and false` | `false` |
| `UNKNOWN and true` | `UNKNOWN` |
| `not UNKNOWN` | `UNKNOWN` |

## Examples

```python
"isnan(loss) or isinf(loss)                        => critical: non-finite loss"
"zscore(loss[50]) > 4                              => error: loss spike {loss:.4f}"
"diff(loss[2]) > 50                                => warning: sudden jump"
"stalled(loss[100]) and step() > 500               => warning: flat for 100 steps"
"mean(grad[20]) > 10 * mean(grad[500])             => warning: exploding gradients"
"increasing(eval.loss[5])                          => warning: overfitting"
"pct_change(lr[2]) < -0.5                          => info: lr dropped"
"no_data(10m)                                      => error: training looks hung"
"age(eval.acc) > 3600                              => warning: no eval for an hour"
"elapsed() > 86400                                 => info: running for a day"
"has(grad_norm) and grad_norm > 1e4                => error: gradient blow-up"
```

## Python builder

The same language with editor completion:

```python
from expr_tracker import M

M.loss > 5                          # loss > 5
M.train.loss > 5                    # train.loss > 5
M["odd name"] > 5                   # `odd name` > 5
M.loss[50].zscore() > 4             # zscore(loss[50]) > 4
M.loss["5m"].mean() > 1             # mean(loss[5m]) > 1
(M.a > 1) & (M.b > 2)               # and
(M.a > 1) | (M.b > 2)               # or
~(M.a > 1)                          # not
M.loss * 2 + 1 > 0                  # arithmetic, both operand orders
```

Use `&` `|` `~`, not `and` `or` `not` — Python cannot overload the keywords.

## Message templates

`{step}`, `{time}`, `{run}`, `{project}`, any metric on the current step, and
`{expr}` — the condition rendered with the values that were observed:

```
diff(m1)=63.2 > 50 or m1=1.2 > 5
```

Format specifiers work (`{loss:.4f}`). An unknown placeholder is left as written
instead of raising.
