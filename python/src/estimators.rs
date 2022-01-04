use numpy::{PyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyFloat;
use rustylasso;

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
    inner: rustylasso::estimators::Lasso<f32>,
}

#[pymethods]
impl Lasso {
    #[new]
    fn new(alpha: PyFloat) -> PyResult<(Self, BaseEstimator)> {
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
        X: PyArray2<f32>,
        y: PyArray1<f32>,
    ) -> PyResult<&PyArray1<f32>> {
        Ok(PyArray::from_array(
            py,
            self.inner.fit(X.as_array(), y.as_array()),
        ))
    }
}
