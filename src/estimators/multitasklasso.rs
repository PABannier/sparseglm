extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix2, OwnedRepr, ViewRepr};

use super::error::{LassoError, Result};
use super::hyperparams::{MultiTaskLassoParams, MultiTaskLassoValidParams};
use super::traits::Fit;
use crate::bcd::block_coordinate_descent;
use crate::datafits::QuadraticMultiTask;
use crate::datasets::{csc_array::CSCArray, DatasetBase};
use crate::penalties::L21;
use crate::solvers::Solver;
use crate::Float;

/// MultiTask Lasso
///
/// The MultiTaskLasso estimator solves a regularized least-square regression problem
/// with a measurement matrix. The problem is regularized using a L21 norm and yields
/// structured sparse solutions.
pub struct MultiTaskLasso<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> MultiTaskLasso<F> {
    /// Create an instance of the Lasso with default parameters
    pub fn params() -> MultiTaskLassoParams<F> {
        MultiTaskLassoParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.coefficients.view()
    }
}

impl<'a, F, D> Fit<'a, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, LassoError>
    for MultiTaskLassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = MultiTaskLasso<F>;
    /// Fits the MultiTaskLasso estimator to a dense design matrix
    fn fit(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = QuadraticMultiTask::default();
        let penalty = L21::new(self.alpha());
        let W = block_coordinate_descent(
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
        Ok(MultiTaskLasso { coefficients: W })
    }
}

impl<'a, F, D> Fit<'a, CSCArray<'a, F>, ArrayBase<D, Ix2>, LassoError>
    for MultiTaskLassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = MultiTaskLasso<F>;
    /// Fits the MultiTask estimator to a sparse design matrix
    fn fit(
        &self,
        dataset: &'a DatasetBase<CSCArray<'a, F>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = QuadraticMultiTask::default();
        let penalty = L21::new(self.alpha());

        let W = block_coordinate_descent(
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
        Ok(MultiTaskLasso { coefficients: W })
    }
}
