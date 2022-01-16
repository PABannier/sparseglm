use super::traits::Fit;
use crate::datasets::{DatasetBase, DesignMatrix, Targets};
use std::error::Error;

/// A set of hyperparameters whose values have not been checked for validity. A reference to the
/// checked hyperparameters can only be obtained after checking has completed. If the
/// `Transformer`, `Fit`, or `FitWith` traits have been implemented on the checked
/// hyperparameters, they will also be implemented on the unchecked hyperparameters with the
/// checking step done automatically.
///
/// The hyperparameter validation done in `check_ref()` and `check()` should be identical.
pub trait ParamGuard {
    /// The checked hyperparameters
    type Checked;
    /// Error type resulting from failed hyperparameter checking
    type Error: Error;

    /// Checks the hyperparameters and returns a reference to the checked hyperparameters if
    /// successful
    fn check_ref(&self) -> Result<&Self::Checked, Self::Error>;

    /// Checks the hyperparameters and returns the checked hyperparameters if successful
    fn check(self) -> Result<Self::Checked, Self::Error>;

    /// Calls `check()` and unwraps the result
    fn check_unwrap(self) -> Self::Checked
    where
        Self: Sized,
    {
        self.check().unwrap()
    }
}

/// Performs checking step and calls `fit` on the checked hyperparameters. If checking failed, the
/// checking error is converted to the original error type of `Fit` and returned.
impl<DM: DesignMatrix, T: Targets, E, P: ParamGuard> Fit<DM, T, E> for P
where
    P::Checked: Fit<DM, T, E>,
    E: Error + From<P::Error>,
{
    type Object = <<P as ParamGuard>::Checked as Fit<DM, T, E>>::Object;

    fn fit(&self, dataset: &DatasetBase<DM, T>) -> Result<Self::Object, E> {
        let checked = self.check_ref()?;
        checked.fit(dataset)
    }
}
