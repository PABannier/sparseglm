extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix2, OwnedRepr, ViewRepr};

use super::error::{LassoError, Result};
use super::hyperparams::{MultiTaskLassoParams, MultiTaskLassoValidParams};
use super::traits::Fit;
use crate::bcd::block_coordinate_descent;
use crate::datafits_multitask::QuadraticMultiTask;
use crate::datasets::{csc_array::CSCArray, DatasetBase};
use crate::penalties_multitask::L21;
use crate::solver_multitask::MultiTaskSolver;
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

impl<F, D> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, LassoError> for MultiTaskLassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = MultiTaskLasso<F>;
    /// Fits the MultiTaskLasso estimator to a dense design matrix
    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let solver = MultiTaskSolver {};
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

impl<F, D> Fit<CSCArray<'_, F>, ArrayBase<D, Ix2>, LassoError> for MultiTaskLassoValidParams<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Object = MultiTaskLasso<F>;
    /// Fits the MultiTask estimator to a sparse design matrix
    fn fit(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        let solver = MultiTaskSolver {};
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
