//! Lossless JSONL history, shared by the SDK and the native web service.
#[path = "history/json.rs"]
mod json;
#[path = "history/notifications.rs"]
mod notifications;
#[path = "history/reader.rs"]
mod reader;
#[path = "history/record.rs"]
mod record;
#[path = "history/series.rs"]
mod series;
#[path = "history/state.rs"]
mod state;
#[cfg(test)]
#[path = "history/tests.rs"]
mod tests;
#[path = "history/writer.rs"]
mod writer;

pub use json::{encode_browser_value, parse_compatible_json, CompatibleJson, JsonObject, Step};
pub use reader::{
    discover_runs, segment_paths, visit_rows, visit_rows_from, HistoryQuery, RawRow,
    SegmentedReader,
};
pub use series::NativeMetricSeries;

use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<record::NativeRecord>()?;
    module.add_function(wrap_pyfunction!(record::history_metric_values, module)?)?;
    reader::register(module)?;
    writer::register(module)?;
    state::register(module)?;
    series::register(module)?;
    notifications::register(module)?;
    Ok(())
}
