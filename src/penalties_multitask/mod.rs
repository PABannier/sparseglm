extern crate ndarray;

use ndarray::{s, Array1, ArrayView1, ArrayView2, Axis};

use super::Float;
use crate::helpers::prox::block_soft_thresholding;

#[cfg(test)]
mod tests;

pub trait PenaltyMultiTask<T: Float> {
    fn value(&self, W: ArrayView2<T>) -> T;
    fn prox_op(&self, value: ArrayView1<T>, stepsize: T) -> Array1<T>;
    fn subdiff_distance(
        &self,
        W: ArrayView2<T>,
        grad: ArrayView2<T>,
        ws: ArrayView1<usize>,
    ) -> (Array1<T>, T);
}

/// L21 penalty
///

pub struct L21<T> {
    alpha: T,
}

impl<T: Float> L21<T> {
    // Constructor
    pub fn new(alpha: T) -> Self {
        L21 { alpha }
    }
}

impl<T: 'static + Float> PenaltyMultiTask<T> for L21<T> {
    /// Gets the current value of the penalty
    fn value(&self, W: ArrayView2<T>) -> T {
        self.alpha * W.map_axis(Axis(1), |Wj| (Wj.dot(&Wj).sqrt())).sum()
    }
    /// Computes the value of the proximal operator
    fn prox_op(&self, value: ArrayView1<T>, stepsize: T) -> Array1<T> {
        block_soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        W: ArrayView2<T>,
        grad: ArrayView2<T>,
        ws: ArrayView1<usize>,
    ) -> (Array1<T>, T) {
        let ws_size = ws.len();
        let n_tasks = W.shape()[1];
        let mut subdiff_dist = Array1::zeros(ws_size);
        let mut max_subdiff_dist = T::neg_infinity();
        for (idx, &j) in ws.iter().enumerate() {
            if W.slice(s![j, ..]).map(|&w| w.abs()).sum() == T::zero() {
                let norm_grad_j = grad.slice(s![idx, ..]).map(|&x| x * x).sum().sqrt();
                subdiff_dist[idx] = T::max(T::zero(), norm_grad_j - self.alpha);
            } else {
                let mut res = Array1::<T>::zeros(n_tasks);
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
