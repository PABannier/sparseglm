extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix1, Ix2, OwnedRepr, ViewRepr};

use super::traits::Fit;
use crate::cd::coordinate_descent;
use crate::datafits::Quadratic;
use crate::datasets::{csc_array::CSCArray, DatasetBase};
use crate::hyperparams::{LassoParams, LassoValidParams};
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

impl<F, D> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>> for LassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = Lasso<F>;
    /// Fits the Lasso estimator to a dense design matrix
    fn fit(&mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>) -> Self::Object {
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
        Lasso { coefficients: w }
    }
}

impl<F, D> Fit<CSCArray<'_, F>, ArrayBase<D, Ix1>> for LassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = Lasso<F>;
    /// Fits the Lasso estimator to a sparse design matrix
    fn fit(&mut self, dataset: &DatasetBase<CSCArray<F>, ArrayBase<D, Ix1>>) -> Self::Object {
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
        Lasso { coefficients: w }
    }
}