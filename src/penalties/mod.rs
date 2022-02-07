extern crate ndarray;

use ndarray::{Array1, ArrayBase, ArrayView1, Ix1, OwnedRepr};

use super::Float;
use crate::helpers::prox::soft_thresholding;

#[cfg(test)]
mod tests;

pub trait Penalty<F: Float> {
    fn value(&self, w: ArrayView1<F>) -> F;
    fn prox_op(&self, value: F, step_size: F) -> F;
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F);
}

/// L1 penalty
///

pub struct L1<F: Float> {
    alpha: F,
}

impl<F: Float> L1<F> {
    // Constructor
    pub fn new(alpha: F) -> Self {
        L1 { alpha }
    }
}

impl<F: Float> Penalty<F> for L1<F> {
    /// Gets the current value of the penalty
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha * w.fold(F::zero(), |sum, &x| sum + x.abs())
    }
    /// Computes the value of the proximal operator
    fn prox_op(&self, value: F, stepsize: F) -> F {
        soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F) {
        let ws_size = ws.len();
        let mut subdiff_dist = Array1::<F>::zeros(ws_size);
        let mut max_subdiff_dist = F::neg_infinity();
        for (idx, &j) in ws.iter().enumerate() {
            if w[j] == F::zero() {
                subdiff_dist[idx] = F::max(F::zero(), grad[idx].abs() - self.alpha);
            } else {
                subdiff_dist[idx] = (-grad[idx] - w[j].signum() * self.alpha).abs();
            }

            if subdiff_dist[idx] > max_subdiff_dist {
                max_subdiff_dist = subdiff_dist[idx];
            }
        }
        (subdiff_dist, max_subdiff_dist)
    }
}
