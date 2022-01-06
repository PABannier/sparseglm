use pyo3::prelude::*;

#[allow(non_snake_case)]
mod estimators;

#[pymodule]
fn rustylassopy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<estimators::LassoWrapperF32>()?;
    m.add_class::<estimators::LassoWrapperF64>()?;
    Ok(())
}
