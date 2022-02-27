extern crate ndarray;

use ndarray::{Array1, ArrayBase, ArrayView1, Data, Dimension, Ix1, Ix2};

use super::error::{LassoError, Result};
use super::hyperparams::{LassoParams, LassoValidParams};
use super::traits::Fit;
use crate::bcd::block_coordinate_descent;
use crate::cd::coordinate_descent;
use crate::datafits::Quadratic;
use crate::datafits_multitask::QuadraticMultiTask;
use crate::datasets::{csc_array::CSCArray, AsMultiTargets, AsSingleTargets, DatasetBase};
use crate::penalties::{L1, MCP};
use crate::penalties_multitask::{BlockMCP, L21};
use crate::solver::Solver;
use crate::solver_multitask::MultiTaskSolver;
use crate::Float;

/// Lasso
///
/// The Lasso estimator solves a regularized least-square regression problem.
/// The L1-regularization used yields sparse solutions. In the Multi-Task case,
/// the problem is regularized using a L21 norm and yields structured sparse
/// solutions.
pub struct Lasso<F, S: Data<Elem = F>, I: Dimension> {
    coefficients: ArrayBase<S, I>,
}

impl<F: Float, S: Data<Elem = F>, I: Dimension> Lasso<F, S, I> {
    /// Creates an instance of the Lasso with default parameters
    pub fn params() -> LassoParams<F> {
        LassoParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<S, I> {
        self.coefficients.view()
    }
}

impl<F: Float, S: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, LassoError> for LassoValidParams<F>
{
    type Object = Lasso<F, S, Ix1>;
    /// Fits the Lasso estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
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

impl<F: Float, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, LassoError>
    for LassoValidParams<F>
{
    type Object = Lasso<F, "he", Ix1>;
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

impl<F: Float, S: Data<Elem = F>, T: AsMultiTargets<Elem = F>> Fit<ArrayBase<S, Ix2>, T, LassoError>
    for LassoValidParams<F>
{
    type Object = Lasso<F, S, Ix2>;

    /// Fits the MultiTaskLasso estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
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
        Ok(Lasso { coefficients: W })
    }
}

impl<F: Float, T: AsMultiTargets<Elem = F>> Fit<CSCArray<'_, F>, T, LassoError>
    for LassoValidParams<F>
{
    type Object = Lasso<F, "he", Ix2>;

    /// Fits the MultiTask estimator to a sparse design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>) -> Result<Self::Object> {
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
        Ok(Lasso { coefficients: W })
    }
}

/// MCP Regressor
///
/// The Minimax Concave Penalty (MCP) estimator yields sparser solution than the
/// Lasso thanks to a non-convex penalty. This mitigates the intrinsic Lasso bias
/// and offers sparser solutions.
pub struct MCPEstimator<F> {
    coefficients: Array1<F>,
}

impl<F: Float> MCPEstimator<F> {
    /// Creates an instance of the Lasso with default parameters
    pub fn params() -> MCParams<F> {
        MCParams::new()
    }

    pub fn coefficients(&self) -> ArrayView1<F> {
        self.coefficients.view()
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<D, Ix2>, T, LassoError> for MCPValidParams<F>
{
    type Object = MCPEstimator<F>;

    /// Fits the MCP estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = Quadratic::default();
        let penalty = MCP::new(self.alpha(), self.gamma());

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
        Ok(MCPEstimator { coefficients: w })
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, LassoError>
    for MCPValidParams<F>
{
    type Object = MCPEstimator<F>;

    /// Fits the MCP estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<F>, T>) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = Quadratic::default();
        let penalty = MCP::new(self.alpha(), self.gamma());

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
        Ok(MCPEstimator { coefficients: w })
    }
}
