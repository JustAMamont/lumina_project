// Only import PyO3 if we are building the Python extension
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;

pub mod types;
pub mod utils;
pub mod reader;

// The engine contains all the Python interaction logic.
// We exclude it from the standalone binary build to prevent linking errors.
#[cfg(feature = "extension-module")]
mod engine;

// The module initialization function is only needed for Python
#[cfg(feature = "extension-module")]
#[pymodule]
fn lumina_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use engine::LuminaEngine;
    use reader::read_binary_log; 
    
    m.add_class::<LuminaEngine>()?;
    m.add_function(wrap_pyfunction!(read_binary_log, m)?)?;
    Ok(())
}