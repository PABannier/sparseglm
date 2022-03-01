use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rustylasso::{
    datasets::{csc_array::CSCArray, DenseDatasetView, SparseDataset},
    estimators::hyperparams::MCParams,
    estimators::traits::Fit,
};

#[pyclass]
pub struct MCPWrapper {
    inner: MCParams<f64>,
}

#[pymethods]
impl MCPWrapper {
    #[new]
    fn new(
        alpha: f64,
        gamma: f64,
        max_iterations: usize,
        max_epochs: usize,
        tolerance: f64,
        p0: usize,
        use_acceleration: bool,
        k: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let _estimator = MCParams::new()
            .alpha(alpha)
            .gamma(gamma)
            .max_iterations(max_iterations)
            .max_epochs(max_epochs)
            .tolerance(tolerance)
            .p0(p0)
            .use_acceleration(use_acceleration)
            .K(k)
            .verbose(verbose);
        Ok(MCPWrapper { inner: _estimator })
    }

    unsafe fn fit<'py>(
        &mut self,
        py: Python<'py>,
        x: &PyArray2<f64>,
        y: &PyArray1<f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let dataset = DenseDatasetView::from((x.as_array(), y.as_array()));
        let _estimator = self.inner.fit(&dataset).unwrap();
        Ok(PyArray::from_array(py, &_estimator.coefficients()))
    }

    unsafe fn fit_sparse<'py>(
        &mut self,
        py: Python<'py>,
        data: &PyArray1<f64>,
        indices: &PyArray1<i32>,
        indptr: &PyArray1<i32>,
        y: &PyArray1<f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let x = CSCArray::new(data.as_array(), indices.as_array(), indptr.as_array());
        let dataset = SparseDataset::from((x, y.as_array()));
        let _estimator = self.inner.fit(&dataset).unwrap();
        Ok(PyArray::from_array(py, &_estimator.coefficients()))
    }
}
