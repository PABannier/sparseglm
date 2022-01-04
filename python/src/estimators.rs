use numpy::ndarray::{PyArray1, PyArray2};
use pyo3::prelude::*;

use rustylasso::estimators::*;

#[pyclass(module = "RustyLasso.estimators")]
pub struct BaseEstimator {}

#[pymethods]
impl BaseEstimator {
    #[new]
    pub fn new() -> Self {
        BaseEstimator {}
    }
}

/// __init__(self)
///
/// Lasso estimator
#[pyclass(extends=BaseEstimator, module="RustyLasso.estimators")]
pub struct Lasso {
    inner: RustyLasso::estimators::Lasso,
}

#[pymethods]
impl Lasso {
    #[new]
    fn new(alpha: &PyFloat) -> PyResult<(Self, BaseEstimator)> {
        let estimator = RustyLasso::estimators::Lasso::new(alpha);

        Ok((Lasso { inner: estimator }, BaseEstimator::new()))
    }

    /// fit(self, X, y)
    ///
    /// Fits the estimator to the data
    ///
    /// Parameters
    /// ----------
    /// X:  PyArray2
    ///    The design matrix
    /// y:  PyArray1
    ///    Measurement vector
    fn fit<'py>(&self, py: Python<'py>, X: PyArray2<f64>, y: PyArray1<f64>) -> PyResult<PyArray1> {
        Ok(self.inner.fit(X, y))
    }

    /// fit_sparse(self, X, y)
    ///
    /// Fits the estimator to a sparse design matrix
    ///
    /// Parameters
    /// ----------
    /// X: CSRArray
    ///     The sparse design matrix
    /// y: PyArray1
    ///     Measurement vector
    fn fit_sparse<'py>(
        &self,
        py: Python<'py>,
        X: PyCSRArray<f64>,
        y: PyArray1<f64>,
    ) -> PyResult<PyArray1> {
        Ok(self.inner.fit_sparse(X, y))
    }

    /// get_params(self)
    /// Get parameters for this estimator.
    ///
    /// Returns
    /// -------
    /// params : mapping of string to any
    ///          Parameter names mapped to their values.
    fn get_params(&self) -> PyResult<SolverParams> {
        Ok(self.inner.params.clone())
    }
}
