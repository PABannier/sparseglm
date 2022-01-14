extern crate ndarray;

use ndarray::{s, Array1, ArrayBase, ArrayView1, Axis, Dimension, Ix1, Ix2, OwnedRepr, ViewRepr};

use super::Float;
use crate::helpers::prox::{block_soft_thresholding, soft_thresholding};

#[cfg(test)]
mod tests;

pub trait Penalty<F, I>
where
    F: Float,
    I: Dimension,
{
    type Input;
    type Output;

    fn value(&self, w: ArrayBase<ViewRepr<&F>, I>) -> F;
    fn prox_op(&self, value: Self::Input, step_size: F) -> Self::Output;
    fn subdiff_distance(
        &self,
        w: ArrayBase<ViewRepr<&F>, I>,
        grad: ArrayBase<ViewRepr<&F>, I>,
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

impl<F: Float> Penalty<F, Ix1> for L1<F> {
    type Input = F;
    type Output = F;

    /// Gets the current value of the penalty
    fn value(&self, w: ArrayBase<ViewRepr<&F>, Ix1>) -> F {
        self.alpha * w.map(|x| (*x).abs()).sum()
    }
    /// Computes the value of the proximal operator
    fn prox_op(&self, value: Self::Input, stepsize: F) -> Self::Output {
        soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        w: ArrayBase<ViewRepr<&F>, Ix1>,
        grad: ArrayBase<ViewRepr<&F>, Ix1>,
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

impl<F: Float> Penalty<F, Ix2> for L21<F> {
    type Input = ArrayBase<ViewRepr<&'static F>, Ix1>;
    type Output = ArrayBase<OwnedRepr<F>, Ix1>;

    /// Gets the current value of the penalty
    fn value(&self, W: ArrayBase<ViewRepr<&F>, Ix2>) -> F {
        self.alpha * W.map_axis(Axis(1), |Wj| (Wj.dot(&Wj).sqrt())).sum()
    }
    /// Computes the value of the proximal operator
    fn prox_op(&self, value: Self::Input, stepsize: F) -> Self::Output {
        block_soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        W: ArrayBase<ViewRepr<&F>, Ix2>,
        grad: ArrayBase<ViewRepr<&F>, Ix2>,
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
