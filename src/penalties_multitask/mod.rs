extern crate ndarray;

use ndarray::{s, Array1, ArrayBase, ArrayView1, ArrayView2, Axis, Ix1, OwnedRepr};

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
    fn prox_op(&self, value: ArrayView1<F>, stepsize: F) -> ArrayBase<OwnedRepr<F>, Ix1> {
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
                let norm_grad_j = grad
                    .slice(s![idx, ..])
                    .fold(F::zero(), |sum, &x| sum + x * x)
                    .sqrt();
                subdiff_dist[idx] = F::max(F::zero(), norm_grad_j - self.alpha);
            } else {
                let norm_W_j = W
                    .slice(s![j, ..])
                    .fold(F::zero(), |sum, &wj| sum + wj * wj)
                    .sqrt();
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

/// Block MCP penalty
///

pub struct BlockMCP<F: Float> {
    alpha: F,
    gamma: F,
}

impl<F: Float> BlockMCP<F> {
    /// Constructor
    ///
    pub fn new(alpha: F, gamma: F) -> Self {
        BlockMCP { alpha, gamma }
    }
}

impl<F: Float> PenaltyMultiTask<F> for BlockMCP<F> {
    /// Gets the current value of the penalty
    fn value(&self, W: ArrayView2<F>) -> F {
        // With W_j the j-th row of W, compute
        // pen(||W_j||) = alpha * ||W_j|| - ||W_j||^2 / (2 * gamma)
        //                if ||W_j|| =< gamma * alpha
        //              = gamma * alpha ** 2 / 2
        //                if ||W_j|| > gamma * alpha
        // value = sum_{j=1}^{n_features} pen(||W_j||)
        let cast2 = F::cast(2.);
        let norm_rows = W.map_axis(Axis(1), |Wj| (Wj.dot(&Wj).sqrt()));
        let s0: Vec<bool> = norm_rows
            .iter()
            .map(|&nrm| nrm < self.alpha * self.gamma)
            .collect();
        norm_rows
            .iter()
            .zip(s0)
            .map(|(&nrm_j, s0j)| {
                if s0j {
                    return self.alpha * nrm_j - nrm_j.powi(2) / (cast2 * self.gamma);
                }
                return self.gamma * self.alpha.powi(2) / cast2;
            })
            .sum()
    }

    /// Proximal operator
    fn prox_op(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F> {
        // tau = self.alpha * stepsize
        // g = self.gamma / stepsize
        // norm_value = norm(value)
        // if norm_value <= tau:
        //     return np.zeros_like(value)
        // if norm_value > g * tau:
        //     return value
        // return (1 - tau / norm_value) * value / (1. - 1./g)
        let cast1 = F::cast(1.);
        let tau = self.alpha * stepsize;
        let g = self.gamma / stepsize;
        let norm_value = value.map(|wj| wj.powi(2)).sum().sqrt();
        if norm_value <= tau {
            return Array1::<F>::zeros(value.len());
        } else if norm_value > g * tau {
            return value.to_owned();
        } else {
            Array1::from_vec(
                value
                    .iter()
                    .map(|&value_j| (cast1 - tau / norm_value) * value_j / (cast1 - cast1 / g))
                    .collect(),
            )
        }
    }

    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let ws_size = ws.len();
        let n_tasks = W.shape()[1];
        let mut subdiff_dist = Array1::<F>::zeros(ws_size);
        let mut max_subdiff_dist = F::neg_infinity();
        for (idx, &j) in ws.iter().enumerate() {
            let W_j = W.slice(s![j, ..]);
            let norm_W_j = W_j.map(|wj| wj.powi(2)).sum().sqrt();

            // if not np.any(W[j]):
            //     # distance of -grad_j to alpha * unit ball
            //     norm_grad_j = norm(grad[idx])
            //     subdiff_dist[idx] = max(0, norm_grad_j - self.alpha)
            // elif norm_Wj < self.alpha * self.gamma:
            //     # distance of -grad_j to alpha * W[j] / ||W_j|| -  W[j] / gamma
            //     subdiff_dist[idx] = norm(
            //         grad[idx] + self.alpha * W[j]/norm_Wj - W[j] / self.gamma)
            // else:
            //     # distance of -grad to 0
            //     subdiff_dist[idx] = norm(grad[idx])

            if W_j.map(|wij| wij.abs()).sum() == F::zero {
                let norm_grad_j = grad.slice(s![idx, ..]);
                subdiff_dist[idx] = F::max(F::zero(), norm_grad_j - self.alpha);
            } else if norm_W_j < self.alpha * self.gamma {
                subdiff_dist[idx] = 
            }
        }
    }
}
