extern crate ndarray;

use ndarray::{
    s, Array1, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Data, Ix1, OwnedRepr,
};

use super::Float;
use crate::helpers::prox::block_soft_thresholding;

#[cfg(test)]
mod tests;

pub trait PenaltyMultiTask<F, D>
where
    F: Float,
    D: Data<Elem = F>,
{
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

pub struct L21<F>
where
    F: Float,
{
    alpha: F,
}

impl<F> L21<F>
where
    F: Float,
{
    // Constructor
    pub fn new(alpha: F) -> Self {
        L21 { alpha }
    }
}

impl<F, D> PenaltyMultiTask<F, D> for L21<F>
where
    F: 'static + Float,
    D: Data<Elem = F>,
{
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
            if W.slice(s![j, ..]).map(|&w| w.abs()).sum() == F::zero() {
                let norm_grad_j = grad.slice(s![idx, ..]).map(|&x| x * x).sum().sqrt();
                subdiff_dist[idx] = F::max(F::zero(), norm_grad_j - self.alpha);
            } else {
                let mut res = Array1::<F>::zeros(n_tasks);
                let norm_W_j = W.slice(s![j, ..]).map(|&wj| wj * wj).sum().sqrt();
                for t in 0..n_tasks {
                    res[t] = grad[[idx, t]] + self.alpha * W[[j, t]] / norm_W_j;
                }
                subdiff_dist[idx] = res.map(|&x| x * x).sum().sqrt();
            }

            if subdiff_dist[idx] > max_subdiff_dist {
                max_subdiff_dist = subdiff_dist[idx];
            }
        }
        (subdiff_dist, max_subdiff_dist)
    }
}
