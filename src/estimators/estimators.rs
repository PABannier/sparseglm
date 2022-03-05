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
use crate::Float;

/// The Lasso estimator
///
/// The Lasso estimator solves a regularized least-square regression problem.
/// The L1-regularization used yields sparse solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct Lasso<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix1>,
}

impl<F: Float> Lasso<F> {
    /// This method instantiates a Lasso estimator with default parameters
    /// for the coordinate descent solvers.
    pub fn params() -> LassoParams<F> {
        LassoParams::new()
    }

    /// This method is a getter for the coefficients vector.
    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.coefficients.view()
    }
}

/// This implements the coordinate descent optimization procedure for single-task
/// problem and dense design matrices.
impl<F: Float, S: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for LassoValidParams<F>
{
    /// If successful, the output of the coordinate descent solver is an instance
    /// of [`Lasso`] containing the fitted coefficients.
    type Object = Lasso<F>;

    /// This method fits a [`Lasso`] instance to a dataset with a dense design
    /// matrix.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let mut datafit = Quadratic::new();
        let penalty = L1::new(self.alpha());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
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

/// This implements the coordinate descent optimization procedure for single-task
/// problem and sparse design matrices.
impl<F: Float, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for LassoValidParams<F>
{
    /// If successful, the output of the coordinate descent solver is an instance
    /// of [`Lasso`] containing the fitted coefficients.
    type Object = Lasso<F>;

    /// This method fits a [`Lasso`] instance to a dataset with a sparse design
    /// matrix.
    fn fit(&self, dataset: &DatasetBase<CSCArray<F>, T>) -> Result<Self::Object> {
        let mut datafit = Quadratic::new();
        let penalty = L1::new(self.alpha());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
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

/// The MultiTaskLasso estimator
///
/// The MultiTaskLasso estimator solves a regularized multi-task least-squares
/// regression problem. The L21-regularization used yields sparse solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskLasso<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> MultiTaskLasso<F> {
    /// This method instantiates a [`MultiTaskLasso`] estimator with default
    /// parameters for the block coordinate descent solvers.
    pub fn params() -> MultiTaskLassoParams<F> {
        MultiTaskLassoParams::new()
    }

    /// This method is a getter for the coefficients matrix.
    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.coefficients.view()
    }
}

/// This implements the block coordinate descent optimization procedure for multi-task
/// problem and dense design matrices.
impl<F: Float, S: Data<Elem = F>, T: AsMultiTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for MultiTaskLassoValidParams<F>
{
    /// If successful, the output of the block coordinate descent solver is an
    /// instance of [`MultiTaskLasso`] containing the fitted coefficients.
    type Object = MultiTaskLasso<F>;

    /// This method fits a [`MultiTaskLasso`] instance to a dataset with a
    /// sparse design matrix.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let mut datafit = QuadraticMultiTask::new();
        let penalty = L21::new(self.alpha());
        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
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

/// This implements the block coordinate descent optimization procedure for
/// multi-task problem and sparse design matrices.
impl<F: Float, T: AsMultiTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for MultiTaskLassoValidParams<F>
{
    /// If successful, the output of the block coordinate descent solver is an
    /// instance of [`MultiTaskLasso`] containing the fitted coefficients.
    type Object = MultiTaskLasso<F>;

    /// This method fits a [`MultiTaskLasso`] instance to a dataset with a dense
    /// sparse matrix.
    fn fit(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>) -> Result<Self::Object> {
        let mut datafit = QuadraticMultiTask::new();
        let penalty = L21::new(self.alpha());

        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
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
/// [`Lasso`] thanks to a non-convex penalty. This mitigates the intrinsic
/// [`Lasso`] bias and offers sparser solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct MCPEstimator<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix1>,
}

impl<F: Float> MCPEstimator<F> {
    /// This method instantiates a [`MCPEstimator`] with default parameters
    /// for the coordinate descent optimization procedure.
    pub fn params() -> MCParams<F> {
        MCParams::new()
    }

    /// This is a getter method for the coefficients of the estimator.
    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.coefficients.view()
    }
}

/// This implements the coordinate descent optimization procedure for single-task
/// problems and dense design matrices.
impl<F: Float, S: Data<Elem = F>, T: AsSingleTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for MCPValidParams<F>
{
    /// If successful, the output of the coordinate descent solver is an
    /// instance of [`MCPEstimator`] containing the fitted coefficients.
    type Object = MCPEstimator<F>;

    /// This method fits a [`MCPEstimator`] instance to a dataset with a dense
    /// design matrix.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let mut datafit = Quadratic::new();
        let penalty = MCP::new(self.alpha(), self.gamma());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
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

/// This implements the coordinate descent optimization procedure for single-task
/// problems and sparse design matrices.
impl<F: Float, T: AsSingleTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for MCPValidParams<F>
{
    /// If successful, the output of the coordinate descent solver is an
    /// instance of [`MCPEstimator`] containing the fitted coefficients.
    type Object = MCPEstimator<F>;

    /// This method fits a [`MCPEstimator`] instance to a dataset with a sparse
    /// design matrix.
    fn fit(&self, dataset: &DatasetBase<CSCArray<F>, T>) -> Result<Self::Object> {
        let mut datafit = Quadratic::new();
        let penalty = MCP::new(self.alpha(), self.gamma());

        let w = coordinate_descent(
            dataset,
            &mut datafit,
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
/// The Block Minimax Concave Penalty (MCP) estimator yields sparser solution
/// than the [`MultiTaskLasso`] thanks to a block non-convex penalty.
/// This mitigates the intrinsic [`MultiTaskLasso`] bias and offers sparser
/// solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockMCPEstimator<F> {
    coefficients: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> BlockMCPEstimator<F> {
    /// This method instantiates a [`BlockMCPEstimator`] with default block
    /// coordinate descent parameters.
    pub fn params() -> BlockMCParams<F> {
        BlockMCParams::new()
    }

    /// This method is a getter for the coefficients of the estimator.
    pub fn coefficients(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.coefficients.view()
    }
}

/// This implements the block coordinate descent optimization procedure for
/// multi-task problems and dense design matrices.
impl<F: Float, S: Data<Elem = F>, T: AsMultiTargets<Elem = F>>
    Fit<ArrayBase<S, Ix2>, T, EstimatorError> for BlockMCPValidParams<F>
{
    /// If successful, the output of the block coordinate descent solver is an
    /// instance of [`BlockMCPEstimator`] containing the fitted coefficients.
    type Object = BlockMCPEstimator<F>;

    /// This method fits a [`BlockMCPEstimator`] instance to a dataset with a
    /// dense design matrix.
    fn fit(&self, dataset: &DatasetBase<ArrayBase<S, Ix2>, T>) -> Result<Self::Object> {
        let mut datafit = QuadraticMultiTask::new();
        let penalty = BlockMCP::new(self.alpha(), self.gamma());

        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
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

/// This implements the block coordinate descent optimization procedure for
/// multi-task problems and sparse design matrices.
impl<F: Float, T: AsMultiTargets<Elem = F>> Fit<CSCArray<'_, F>, T, EstimatorError>
    for BlockMCPValidParams<F>
{
    /// If successful, the output of the block coordinate descent solver is an
    /// instance of [`BlockMCPEstimator`] containing the fitted coefficients.
    type Object = BlockMCPEstimator<F>;

    /// This method fits a [`BlockMCPEstimator`] instance to a dataset with a
    /// sparse design matrix.
    fn fit(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>) -> Result<Self::Object> {
        let mut datafit = QuadraticMultiTask::new();
        let penalty = BlockMCP::new(self.alpha(), self.gamma());

        let W = block_coordinate_descent(
            dataset,
            &mut datafit,
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
