extern crate ndarray;

use ndarray::{s, Array1, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Ix1, OwnedRepr};

use super::Float;
use crate::helpers::prox::block_soft_thresholding;

#[cfg(test)]
mod tests;

pub trait PenaltyMultiTask<F: Float> {
    fn value(&self, W: ArrayView2<F>) -> F;
    fn prox_op(&self, value: ArrayView1<F>, stepsize: F) -> ArrayBase<OwnedRepr<F>, Ix1>;
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F);
}

/// L21 penalty
///

pub struct L21<F: Float> {
    alpha: F,
}

impl<F: Float> L21<F> {
    // Constructor
    pub fn new(alpha: F) -> Self {
        L21 { alpha }
    }
}

impl<F: 'static + Float> PenaltyMultiTask<F> for L21<F> {
    /// Gets the current value of the penalty
    fn value(&self, W: ArrayView2<F>) -> F {
        self.alpha * W.map_axis(Axis(1), |Wj| (Wj.dot(&Wj).sqrt())).sum()
    }
    /// Computes the value of the proximal operator
    fn prox_op(&self, value: ArrayView<F, Ix1>, stepsize: F) -> ArrayBase<OwnedRepr<F>, Ix1> {
        block_soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F) {
        let ws_size = ws.len();
        let n_tasks = W.shape()[1];
        let mut subdiff_dist = Array1::<F>::zeros(ws_size);
        let mut max_subdiff_dist = F::neg_infinity();
        for (idx, &j) in ws.iter().enumerate() {
            if W.slice(s![j, ..]).fold(F::zero(), |sum, &w| sum + w.abs()) == F::zero() {
                let norm_grad_j = grad.slice(s![idx, ..]).fold(F::zero(), |sum, &x| sum + x * x).sqrt();
                subdiff_dist[idx] = F::max(F::zero(), norm_grad_j - self.alpha);
            } else {
                let norm_W_j = W.slice(s![j, ..]).fold(F::zero(), |sum, &wj| sum + wj * wj).sqrt();
                let mut res = Array1::<F>::zeros(n_tasks);
                for t in 0..n_tasks {
                    res[t] = grad[[idx, t]] + self.alpha * W[[j, t]] / norm_W_j;
                }
                subdiff_dist[idx] = res.fold(F::zero(), |sum, &x| sum + x * x).sqrt();
            }

            if subdiff_dist[idx] > max_subdiff_dist {
                max_subdiff_dist = subdiff_dist[idx];
            }
        }
        (subdiff_dist, max_subdiff_dist)
    }
}
