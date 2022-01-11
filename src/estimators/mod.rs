extern crate ndarray;

use ndarray::{Array, ArrayBase, Data, Dimension, Ix1, Ix2};

use super::Float;
use crate::datafits::{Quadratic, QuadraticMultiTask};
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};
use crate::penalties::L1;
use crate::penalties_multitask::L21;
use crate::solver::solver;
use crate::solver_multitask::solver_multitask;

#[cfg(test)]
mod tests;

/// Solver parameters
///
/// This block gives default parameters to the coordinate descent solver.
pub struct SolverParams<F> {
    tol: F,
    max_epochs: usize,
    max_iter: usize,
    p0: usize,
    use_accel: bool,
    K: usize,
    verbose: bool,
}

impl<F: Float> Default for SolverParams<F> {
    /// Create an instance with default parameters
    fn default() -> SolverParams<F> {
        SolverParams {
            tol: F::from(1e-9).unwrap(),
            max_epochs: 1000,
            max_iter: 50,
            p0: 10,
            use_accel: true,
            K: 5,
            verbose: true,
        }
    }
}

impl<F: Float> SolverParams<F> {
    /// Create a new instance
    pub fn new(
        max_epochs: usize,
        max_iter: usize,
        p0: usize,
        tol: F,
        K: usize,
        use_accel: bool,
        verbose: bool,
    ) -> Self {
        SolverParams {
            max_epochs,
            max_iter,
            p0,
            tol,
            K,
            use_accel,
            verbose,
        }
    }
}

/// Fit trait
///

pub trait Fit<F, D, DM, T, I>
where
    F: Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    I: Dimension,
{
    fn fit(&self, dataset: &DatasetBase<DM, T>) -> Array<F, I>;
}

/// Lasso
///

pub struct Lasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
    datafit: Quadratic<F, D>,
    penalty: L1<F>,
    params: SolverParams<F>,
}

impl<F, D> Lasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Create new instance
    fn new(alpha: F, params: Option<SolverParams<F>>) -> Self {
        Lasso {
            datafit: Quadratic::default(),
            penalty: L1::new(alpha),
            params: params.unwrap_or(SolverParams::<F>::default()),
        }
    }
}

impl<F, D, T> Fit<F, D, ArrayBase<D, Ix2>, T, Ix1> for Lasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    /// Fits the Lasso estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Array<F, Ix1> {
        let w = solver(
            dataset,
            &mut self.datafit,
            &self.penalty,
            self.params.max_iter,
            self.params.max_epochs,
            self.params.p0,
            self.params.tol,
            self.params.use_accel,
            self.params.K,
            self.params.verbose,
        );
        w
    }
}

impl<F, D, T> Fit<F, D, CSCArray<'_, F>, T, Ix1> for Lasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    /// Fits the Lasso estimator to a sparse design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<F>, T>) -> Array<F, Ix1> {
        let w = solver(
            dataset,
            &mut self.datafit,
            &self.penalty,
            self.params.max_iter,
            self.params.max_epochs,
            self.params.p0,
            self.params.tol,
            self.params.use_accel,
            self.params.K,
            self.params.verbose,
        );
        w
    }
}

/// MultiTask Lasso
///

pub struct MultiTaskLasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
    datafit: QuadraticMultiTask<F, D>,
    penalty: L21<F>,
    params: SolverParams<F>,
}

impl<F, D> MultiTaskLasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Create new instance
    fn new(alpha: F, params: Option<SolverParams<F>>) -> Self {
        MultiTaskLasso {
            datafit: QuadraticMultiTask::default(),
            penalty: L21::new(alpha),
            params: params.unwrap_or(SolverParams::<F>::default()),
        }
    }
}

impl<F, D, T> Fit<F, D, ArrayBase<D, Ix2>, T, Ix2> for MultiTaskLasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    /// Fits the MultiTaskLasso estimator to a dense design matrix
    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Array<F, Ix2> {
        let W = solver_multitask(
            dataset,
            &mut self.datafit,
            &self.penalty,
            self.params.max_iter,
            self.params.max_epochs,
            self.params.p0,
            self.params.tol,
            self.params.use_accel,
            self.params.K,
            self.params.verbose,
        );
        W
    }
}

impl<F, D, T> Fit<F, D, CSCArray<'_, F>, T, Ix2> for MultiTaskLasso<F, D>
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    /// Fits the MultiTask estimator to a sparse design matrix
    fn fit(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>) -> Array<F, Ix2> {
        let W = solver_multitask(
            dataset,
            &mut self.datafit,
            &self.penalty,
            self.params.max_iter,
            self.params.max_epochs,
            self.params.p0,
            self.params.tol,
            self.params.use_accel,
            self.params.K,
            self.params.verbose,
        );
        W
    }
}
