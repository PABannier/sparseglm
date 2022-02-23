extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix1, Ix2, OwnedRepr, ViewRepr};

use super::error::{LassoError, Result};
use super::hyperparams::{LassoParams, LassoValidParams};
use super::traits::Fit;
use crate::cd::coordinate_descent;
use crate::datafits::Quadratic;
use crate::datasets::{csc_array::CSCArray, AsSingleTargets, DatasetBase};
use crate::penalties::L1;
use crate::solver::Solver;
use crate::Float;

/// Lasso
///
/// The Lasso estimator solves a regularized least-square regression problem.
/// The L1-regularization used yields sparse solutions.
pub struct Lasso<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix1>,
}

impl<F: Float> Lasso<F> {
    /// Creates an instance of the Lasso with default parameters
    pub fn params() -> LassoParams<F> {
        LassoParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.coefficients.view()
    }
}

impl<F, D: Data<Elem = F>, T: AsSingleTargets> Fit<ArrayBase<D, Ix2>, T, LassoError>
    for LassoValidParams<F>
{
    type Object = Lasso<F>;
    /// Fits the Lasso estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = Quadratic::default();
        let penalty = L1::new(self.alpha());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
            &solver,
            &penalty,
            self.max_iterations(),
            self.max_epochs(),
            self.p0(),
            self.tolerance(),
            self.use_acceleration(),
            self.K(),
            self.verbose(),
        );
        Ok(Lasso { coefficients: w })
    }
}

impl<F, D: Data<Elem = F>, T: AsSingleTargets> Fit<CSCArray<'_, F>, T, LassoError>
    for LassoValidParams<F>
{
    type Object = Lasso<F>;
    /// Fits the Lasso estimator to a sparse design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<F>, T>) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = Quadratic::default();
        let penalty = L1::new(self.alpha());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
            &solver,
            &penalty,
            self.max_iterations(),
            self.max_epochs(),
            self.p0(),
            self.tolerance(),
            self.use_acceleration(),
            self.K(),
            self.verbose(),
        );
        Ok(Lasso { coefficients: w })
    }
}
