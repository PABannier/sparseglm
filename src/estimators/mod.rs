extern crate ndarray;
extern crate num;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use num::Float;
use std::fmt::Debug;

use crate::datafits::Quadratic;
use crate::datafits_multitask::QuadraticMultiTask;
use crate::penalties::L1;
use crate::penalties_multitask::L21;
use crate::solver::solver;
use crate::solver_multitask::solver_multitask;
use crate::sparse::{CSCArray, MatrixParam};

#[cfg(test)]
mod tests;

pub trait Estimator<'a, T: Float> {
    fn new(alpha: T, params: Option<SolverParams<T>>) -> Self;
    fn fit(&mut self, X: ArrayView2<T>, y: ArrayView1<T>) -> Array1<T>;
    fn fit_sparse(&mut self, X: &'a CSCArray<'a, T>, y: ArrayView1<T>) -> Array1<T>;
}

pub trait MultiTaskEstimator<'a, T: Float> {
    fn new(alpha: T, params: Option<SolverParams<T>>) -> Self;
    fn fit(&mut self, X: ArrayView2<T>, Y: ArrayView2<T>) -> Array2<T>;
    fn fit_sparse(&mut self, X: &'a CSCArray<'a, T>, Y: ArrayView2<T>) -> Array2<T>;
}

pub struct SolverParams<T> {
    tol: T,
    max_epochs: usize,
    max_iter: usize,
    p0: usize,
    use_accel: bool,
    K: usize,
    verbose: bool,
}

impl<T: Float> Default for SolverParams<T> {
    /// Create an instance with default parameters
    fn default() -> SolverParams<T> {
        SolverParams {
            tol: T::from(1e-9).unwrap(),
            max_epochs: 1000,
            max_iter: 50,
            p0: 10,
            use_accel: true,
            K: 5,
            verbose: true,
        }
    }
}

impl<T: Float> SolverParams<T> {
    /// Create a new instance
    pub fn new(
        max_epochs: usize,
        max_iter: usize,
        p0: usize,
        tol: T,
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

/// Lasso
///

pub struct Lasso<T: Float> {
    datafit: Quadratic<T>,
    penalty: L1<T>,
    params: SolverParams<T>,
}

impl<'a, T: 'static + Float + Debug> Estimator<'a, T> for Lasso<T> {
    /// Create new instance
    fn new(alpha: T, params: Option<SolverParams<T>>) -> Self {
        Lasso {
            datafit: Quadratic::default(),
            penalty: L1::new(alpha),
            params: params.unwrap_or(SolverParams::<T>::default()),
        }
    }
    /// Fits an instance of Estimator
    fn fit(&mut self, X: ArrayView2<T>, y: ArrayView1<T>) -> Array1<T> {
        let w = solver(
            MatrixParam::DenseMatrix(X),
            y,
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

    /// Fits an instance of an Estimator to a sparse matrix
    fn fit_sparse(&mut self, X: &'a CSCArray<'a, T>, y: ArrayView1<T>) -> Array1<T> {
        let w = solver(
            MatrixParam::SparseMatrix(X),
            y,
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

pub struct MultiTaskLasso<T: Float> {
    datafit: QuadraticMultiTask<T>,
    penalty: L21<T>,
    params: SolverParams<T>,
}

impl<'a, T: 'static + Float + Debug> MultiTaskEstimator<'a, T> for MultiTaskLasso<T> {
    /// Create new instance
    fn new(alpha: T, params: Option<SolverParams<T>>) -> Self {
        MultiTaskLasso {
            datafit: QuadraticMultiTask::default(),
            penalty: L21::new(alpha),
            params: params.unwrap_or(SolverParams::<T>::default()),
        }
    }
    /// Fits an instance of estimator
    fn fit(&mut self, X: ArrayView2<T>, Y: ArrayView2<T>) -> Array2<T> {
        let W = solver_multitask(
            MatrixParam::DenseMatrix(X),
            Y,
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

    fn fit_sparse(&mut self, X: &'a CSCArray<'a, T>, Y: ArrayView2<T>) -> Array2<T> {
        let W = solver_multitask(
            MatrixParam::SparseMatrix(X),
            Y,
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
