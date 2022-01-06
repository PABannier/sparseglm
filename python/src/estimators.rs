use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rustylasso;
use rustylasso::estimators::Estimator;
use rustylasso::sparse::CSCArray;

#[pyclass]
pub struct LassoWrapperF32 {
    inner: rustylasso::estimators::Lasso<f32>,
}

#[pyclass]
pub struct LassoWrapperF64 {
    inner: rustylasso::estimators::Lasso<f64>,
}

#[pymethods]
impl LassoWrapperF64 {
    #[new]
    fn new(
        alpha: f64,
        max_iter: usize,
        max_epochs: usize,
        tol: f64,
        p0: usize,
        use_accel: bool,
        K: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let params = rustylasso::estimators::SolverParams::new(
            max_epochs, max_iter, p0, tol, K, use_accel, verbose,
        );
        let estimator = rustylasso::estimators::Lasso::new(alpha, Some(params));
        Ok(LassoWrapperF64 { inner: estimator })
    }

    unsafe fn fit<'py>(
        &mut self,
        py: Python<'py>,
        X: &PyArray2<f64>,
        y: &PyArray1<f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray::from_array(
            py,
            &self.inner.fit(X.as_array(), y.as_array()),
        ))
    }

    unsafe fn fit_sparse<'py>(
        &mut self,
        py: Python<'py>,
        X_data: &PyArray1<f64>,
        X_indices: &PyArray1<usize>,
        X_indptr: &PyArray1<usize>,
        y: &PyArray1<f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let X_sparse = CSCArray::new(X_data.to_vec()?, X_indices.to_vec()?, X_indptr.to_vec()?);
        Ok(PyArray::from_array(
            py,
            &self.inner.fit_sparse(&X_sparse, y.as_array()),
        ))
    }
}

#[pymethods]
impl LassoWrapperF32 {
    #[new]
    fn new(
        alpha: f32,
        max_iter: usize,
        max_epochs: usize,
        tol: f32,
        p0: usize,
        use_accel: bool,
        K: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let params = rustylasso::estimators::SolverParams::new(
            max_epochs, max_iter, p0, tol, K, use_accel, verbose,
        );
        let estimator = rustylasso::estimators::Lasso::new(alpha, Some(params));
        Ok(LassoWrapperF32 { inner: estimator })
    }

    unsafe fn fit<'py>(
        &mut self,
        py: Python<'py>,
        X: &PyArray2<f32>,
        y: &PyArray1<f32>,
    ) -> PyResult<&'py PyArray1<f32>> {
        Ok(PyArray::from_array(
            py,
            &self.inner.fit(X.as_array(), y.as_array()),
        ))
    }

    unsafe fn fit_sparse<'py>(
        &mut self,
        py: Python<'py>,
        X_data: &PyArray1<f32>,
        X_indices: &PyArray1<usize>,
        X_indptr: &PyArray1<usize>,
        y: &PyArray1<f32>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let X_sparse = CSCArray::new(X_data.to_vec()?, X_indices.to_vec()?, X_indptr.to_vec()?);
        Ok(PyArray::from_array(
            py,
            &self.inner.fit_sparse(&X_sparse, y.as_array()),
        ))
    }
}
