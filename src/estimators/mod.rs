extern crate ndarray;
extern crate num;

use crate::datafits::Quadratic;
use crate::penalties::L1;
use crate::solver::solver;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num::Float;

#[cfg(tests)]
mod tests;

pub trait Estimator<T: Float> {
    fn new(alpha: T) -> Self;
    fn fit(&self, X: ArrayView2<T>, y: ArrayView1<T>) -> Array1<T>;
}

pub struct SolverParams<T> {
    tol: T,
    max_epochs: usize,
    max_iter: usize,
    p0: usize,
    use_accel: bool,
    k: usize,
    verbose: bool,
}

impl<T: Float> Default for SolverParams<T> {
    /// Create a new instance
    fn default() -> SolverParams<T> {
        SolverParams {
            tol: T::from(1e-9).unwrap(),
            max_epochs: 1000,
            max_iter: 50,
            p0: 10,
            use_accel: true,
            k: 5,
            verbose: true,
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

impl<T: Float + 'static> Estimator<T> for Lasso<T> {
    /// Create new instance
    fn new(alpha: T) -> Self {
        Lasso {
            datafit: Quadratic::default(),
            penalty: L1::new(alpha),
            params: SolverParams::default(),
        }
    }
    /// Fits an instance of Estimator
    fn fit(&self, X: ArrayView2<T>, y: ArrayView1<T>) -> Array1<T> {
        let n_features = X.shape()[1];

        let mut w = Array1::<T>::zeros(n_features);
        let mut Xw = X.dot(&w);

        solver(
            X.view(),
            y.view(),
            &self.datafit,
            &self.penalty,
            &mut w,
            &mut Xw,
            self.params.max_iter,
            self.params.max_epochs,
            self.params.p0,
            self.params.tol,
            self.params.verbose,
        );

        w
    }
}
