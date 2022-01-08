use pyo3::prelude::*;

#[allow(non_snake_case)]
mod lasso;
mod mtl_lasso;

#[pymodule]
fn rustylassopy(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<lasso::LassoWrapperF32>()?;
    m.add_class::<lasso::LassoWrapperF64>()?;
    m.add_class::<mtl_lasso::MultiTaskLassoWrapperF32>()?;
    m.add_class::<mtl_lasso::MultiTaskLassoWrapperF64>()?;
    Ok(())
}
