use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::PyFloat;

use rustylasso::estimators::*;

// type PyCSRArray = (Py<PyArray1<f64>>, Py<PyArray1<i32>>, Py<PyArray1<i32>>);

#[pyclass(module = "rustylasso.estimators")]
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
#[pyclass(extends=BaseEstimator, module="rustylasso.estimators")]
pub struct Lasso {
    inner: rustylasso::estimators::Lasso<f64>,
}

#[pymethods]
impl Lasso {
    #[new]
    fn new(alpha: &PyFloat) -> PyResult<(Self, BaseEstimator)> {
        let estimator = rustylasso::estimators::Lasso::new(alpha);

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
    fn fit<'py>(
        &self,
        py: Python<'py>,
        X: PyArray2<f64>,
        y: PyArray1<f64>,
    ) -> PyResult<PyArray1<f64>> {
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
    // fn fit_sparse<'py>(
    //     &self,
    //     py: Python<'py>,
    //     X: PyCSRArray<f64>,
    //     y: PyArray1<f64>,
    // ) -> PyResult<PyArray1> {
    //     let arr = CSRArray::new(X.1, X.2, X.3);
    //     Ok(self.inner.fit_sparse(X, y))
    // }
}
