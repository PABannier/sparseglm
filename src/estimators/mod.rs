extern crate ndarray;
extern crate num;

use crate::datafits::Quadratic;
use crate::penalties::L1;
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
    alpha: T,
    datafit: Quadratic<T>,
    penalty: L1<T>,
    params: SolverParams<T>,
}

impl<T: Float> Estimator<T> for Lasso<T> {
    /// Create new instance
    fn new(alpha: T) -> Self {
        Lasso {
            alpha,
            datafit: Quadratic::default(),
            penalty: L1::new(alpha),
            params: SolverParams::default(),
        }
    }
    /// Fits an instance of Estimator
    fn fit(&self, X: ArrayView2<T>, y: ArrayView1<T>) -> Array1<T> {
        let coef = Array1::zeros(3);
        coef
    }
}
