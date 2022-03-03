extern crate ndarray;

use ndarray::{ArrayBase, Data, Ix1, Ix2, OwnedRepr, ViewRepr};

use super::error::{EstimatorError, Result};
use super::hyperparams::{
    BlockMCPValidParams, BlockMCParams, LassoParams, LassoValidParams, MCPValidParams, MCParams,
    MultiTaskLassoParams, MultiTaskLassoValidParams,
};
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
#[derive(Debug, Clone, PartialEq)]
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

impl<F: Float, S: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for LassoValidParams<F>
{
    type Object = Lasso<F>;
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
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(Lasso { coefficients: w })
    }
}

impl<F: Float, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
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
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(Lasso { coefficients: w })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskLasso<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> MultiTaskLasso<F> {
    /// Creates an instance of the MultiTaskLasso with default parameters
    pub fn params() -> MultiTaskLassoParams<F> {
        MultiTaskLassoParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.coefficients.view()
    }
}

impl<F: Float, S: Data<Elem = F>, T: AsMultiTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for MultiTaskLassoValidParams<F>
{
    type Object = MultiTaskLasso<F>;

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
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(MultiTaskLasso { coefficients: W })
    }
}

impl<F: Float, T: AsMultiTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for MultiTaskLassoValidParams<F>
{
    type Object = MultiTaskLasso<F>;

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
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(MultiTaskLasso { coefficients: W })
    }
}

/// MCP Regressor
///
/// The Minimax Concave Penalty (MCP) estimator yields sparser solution than the
/// Lasso thanks to a non-convex penalty. This mitigates the intrinsic Lasso bias
/// and offers sparser solutions.

#[derive(Debug, Clone, PartialEq)]
pub struct MCPEstimator<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix1>,
}

impl<F: Float> MCPEstimator<F> {
    /// Creates an instance of the Lasso with default parameters
    pub fn params() -> MCParams<F> {
        MCParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.coefficients.view()
    }
}

impl<F: Float, S: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for MCPValidParams<F>
{
    type Object = MCPEstimator<F>;

    /// Fits the MCP estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let solver = Solver {};
        let mut datafit = Quadratic::default();
        let penalty = MCP::new(self.alpha(), self.gamma());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
            &solver,
            &penalty,
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(MCPEstimator { coefficients: w })
    }
}

impl<F: Float, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
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
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(MCPEstimator { coefficients: w })
    }
}

/// Block MCP Regressor
///
/// The Block Minimax Concave Penalty (MCP) estimator yields sparser solution than the
/// MultiTaskLasso thanks to a block non-convex penalty. This mitigates the intrinsic Lasso bias
/// and offers sparser solutions.

#[derive(Debug, Clone, PartialEq)]
pub struct BlockMCPEstimator<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> BlockMCPEstimator<F> {
    /// Creates an instance of the Lasso with default parameters
    pub fn params() -> BlockMCParams<F> {
        BlockMCParams::new()
    }

    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.coefficients.view()
    }
}

impl<F: Float, S: Data<Elem = F>, T: AsMultiTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for BlockMCPValidParams<F>
{
    type Object = BlockMCPEstimator<F>;

    /// Fits the Block MCP estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let solver = MultiTaskSolver {};
        let mut datafit = QuadraticMultiTask::default();
        let penalty = BlockMCP::new(self.alpha(), self.gamma());

        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
            &solver,
            &penalty,
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(BlockMCPEstimator { coefficients: W })
    }
}

impl<F: Float, T: AsMultiTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for BlockMCPValidParams<F>
{
    type Object = BlockMCPEstimator<F>;

    /// Fits the Block MCP estimator to a sparse design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>) -> Result<Self::Object> {
        let solver = MultiTaskSolver {};
        let mut datafit = QuadraticMultiTask::default();
        let penalty = BlockMCP::new(self.alpha(), self.gamma());

        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
            &solver,
            &penalty,
            self.p0(),
            self.max_iterations(),
            self.max_epochs(),
            self.tolerance(),
            self.K(),
            self.use_acceleration(),
            self.verbose(),
        );
        Ok(BlockMCPEstimator { coefficients: W })
    }
}
