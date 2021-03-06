use crate::datasets::{AsMultiTargets, DatasetBase, DesignMatrix};

/// The [`Fit`] trait allows an estimator to be fitted to a dataset (a combination
/// of design matrix and targets). More formally, the model estimates coefficients
/// that minimizes an objective.
pub trait Fit<DM: DesignMatrix, T: AsMultiTargets, E: std::error::Error> {
    type Object;

    fn fit(&self, dataset: &DatasetBase<DM, T>) -> Result<Self::Object, E>;
}
