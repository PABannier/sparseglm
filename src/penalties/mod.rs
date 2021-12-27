extern crate approx;
extern crate ndarray;
extern crate num;

use crate::solver::soft_thresholding;
use approx::abs_diff_eq;
use ndarray::{Array1, ArrayView1};
use num::Float;

#[cfg(test)]
mod tests;

pub trait Penalty<T: Float> {
    fn value(self, w: ArrayView1<T>) -> T;
    fn prox_op(self, value: T, step_size: T, j: usize) -> T;
    fn subdiff_distance(
        self,
        w: ArrayView1<T>,
        grad: ArrayView1<T>,
        ws: ArrayView1<usize>,
    ) -> Array1<T>;
}

/// L1 penalty
///

pub struct L1Penalty<T> {
    alpha: T,
}

impl<T: Float> Penalty<T> for L1Penalty<T> {
    /// Gets the current value of the penalty
    fn value(self, w: ArrayView1<T>) -> T {
        self.alpha * w.map(|x| T::abs(*x)).sum()
    }
    /// Computes the value of the proximal operator
    fn prox_op(self, value: T, stepsize: T, j: usize) -> T {
        soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        self,
        w: ArrayView1<T>,
        grad: ArrayView1<T>,
        ws: ArrayView1<usize>,
    ) -> Array1<T> {
        let ws_size = ws.len();
        let mut subdiff_dist = Array1::<T>::zeros(ws_size);
        for (idx, &j) in ws.iter().enumerate() {
            if abs_diff_eq!(w[j], T::zero()) {
                subdiff_dist[idx] = T::max(T::zero(), T::abs(grad[idx]) - self.alpha);
            } else {
                subdiff_dist[idx] = T::abs(-grad[idx] - T::signum(w[j]) * self.alpha);
            }
        }
        subdiff_dist
    }
}
