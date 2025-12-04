use pyo3::prelude::*;

mod types;
pub mod utils;
mod drivers;
mod engine;
pub mod reader;

use engine::LuminaEngine;
use reader::read_binary_log; 

/// The Lumina Core Python module.
#[pymodule]
fn lumina_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LuminaEngine>()?;
    m.add_function(wrap_pyfunction!(read_binary_log, m)?)?;
    Ok(())
}