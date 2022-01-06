use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use rustylasso;
use rustylasso::estimators::Estimator;
use rustylasso::sparse::CSCArray;

/// __init__(self)
///
/// Lasso estimator
#[pyclass]
pub struct LassoWrapper {
    inner: rustylasso::estimators::Lasso<f64>,
}

#[pymethods]
impl LassoWrapper {
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
        Ok(LassoWrapper { inner: estimator })
    }

    /// fit(self, X, y)
    ///
    /// Fits the estimator to the data
    ///
    /// Parameters
    /// ----------
    /// X: &PyArray2
    ///    The design matrix
    /// y: &PyArray1
    ///    Measurement vector
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

    /// fit_sparse(self, X, y)
    ///
    /// Fits the estimator to the data for a sparse CSC design matrix
    ///
    /// Parameters
    /// ----------
    /// X: CSCArray
    ///    The design matrix
    /// y: &PyArray1
    ///    Measurement vector
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
