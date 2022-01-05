use pyo3::prelude::*;

#[allow(non_snake_case)]
mod estimators;

#[pymodule]
fn _lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<estimators::BaseEstimator>()?;
    m.add_class::<estimators::LassoWrapper>()?;
    Ok(())
}
