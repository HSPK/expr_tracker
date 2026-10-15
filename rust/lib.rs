use pyo3::prelude::*;

mod alerts;
mod artifacts;
mod history;
mod spans;
mod summary;
mod trace;
mod viewer;
mod viewer_json;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("implementation", "rust")?;
    history::register(module)?;
    alerts::register(module)?;
    artifacts::register(module)?;
    spans::register(module)?;
    summary::register(module)?;
    trace::register(module)?;
    viewer::register(module)?;
    Ok(())
}
