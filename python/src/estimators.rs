use pyo3::prelude::*;
use pyo3::types::PyList;

use RustyLasso::estimators::*;

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
    fn new(alpha: Number) -> PyResult<(Self, BaseEstimator)> {
        let estimator = RustyLasso::estimators::Lasso::new(alpha);

        Ok((Lasso { inner: estimator }, BaseEstimator::new()))
    }

    /// fit(self, X, y)
    ///
    /// Fits the estimator to the data
    ///
    /// Parameters
    /// ----------
    /// X:
    ///    The design matrix
    /// y:
    ///    Measurement vector
    fn fit<'py>(&self, py: Python<'py>, X: ArrayView2<T>, y: ArrayView1<T>) -> PyResult {}
}
