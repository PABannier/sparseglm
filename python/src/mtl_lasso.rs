use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rustylasso::{
    datasets::{csc_array::CSCArray, DenseDatasetView, SparseDatasetView},
    estimators::hyperparams::MultiTaskLassoParams,
    estimators::traits::Fit,
};

#[pyclass]
pub struct MultiTaskLassoWrapper {
    inner: MultiTaskLassoParams<f64>,
}

#[pymethods]
impl MultiTaskLassoWrapper {
    #[new]
    fn new(
        alpha: f64,
        max_iterations: usize,
        max_epochs: usize,
        tolerance: f64,
        p0: usize,
        use_acceleration: bool,
        K: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let _estimator = MultiTaskLassoParams::new()
            .alpha(alpha)
            .max_iterations(max_iterations)
            .max_epochs(max_epochs)
            .tolerance(tolerance)
            .p0(p0)
            .use_acceleration(use_acceleration)
            .K(K)
            .verbose(verbose);
        Ok(MultiTaskLassoWrapper { inner: _estimator })
    }

    unsafe fn fit<'py>(
        &mut self,
        py: Python<'py>,
        X: &PyArray2<f64>,
        Y: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let dataset = DenseDatasetView::from((X.as_array(), Y.as_array()));
        let _estimator = self.inner.fit(&dataset).unwrap();
        Ok(PyArray::from_array(py, &_estimator.coefficients()))
    }

    unsafe fn fit_sparse<'py>(
        &mut self,
        py: Python<'py>,
        X_data: &PyArray1<f64>,
        X_indices: &PyArray1<i32>,
        X_indptr: &PyArray1<i32>,
        Y: &PyArray2<f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let X = CSCArray::new(X_data.as_array(), X_indices.as_array(), X_indptr.as_array());
        let dataset = SparseDatasetView::from((X, Y.as_array()));
        let _estimator = self.inner.fit(&dataset).unwrap();
        Ok(PyArray::from_array(py, &_estimator.coefficients()))
    }
}
