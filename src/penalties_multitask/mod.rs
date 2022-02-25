extern crate ndarray;

use ndarray::{s, Array1, ArrayView1, ArrayView2, Axis};

use super::Float;
use crate::helpers::prox::block_soft_thresholding;

#[cfg(test)]
mod tests;

pub trait PenaltyMultiTask<F: Float> {
    fn value(&self, W: ArrayView2<F>) -> F;
    fn prox_op(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F>;
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F);
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
    fn prox_op(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F> {
        block_soft_thresholding(value, self.alpha * stepsize)
    }
    /// Computes the distance of the gradient to the subdifferential
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist = Array1::from_vec(
            grad.axis_iter(Axis(0))
                .zip(ws)
                .map(|(grad_idx, &j)| {
                    let W_j = W.slice(s![j, ..]);
                    match W_j.iter().any(|&wij| wij == F::zero()) {
                        true => {
                            // let norm_W_j = W_j.iter().map(|&W_ij| W_ij.powi(2)).sum().sqrt();
                            // grad_idx
                            //     .iter()
                            //     .zip(W_j)
                            //     .map(|(&grad_ij, &W_ij)| {
                            //         (grad_ij + self.alpha * W_ij / norm_W_j).powi(2)
                            //     })
                            //     .sum()
                            //     .sqrt()
                            let norm_W_j =
                                W_j.fold(F::zero(), |sum, &W_ij| sum + W_ij.powi(2)).sqrt();
                            grad_idx
                                .iter()
                                .zip(W_j)
                                .fold(F::zero(), |sum, (&grad_ij, &W_ij)| {
                                    sum + (grad_ij + self.alpha * W_ij / norm_W_j).powi(2)
                                })
                                .sqrt()
                        }
                        false => {
                            // let norm_grad_j = grad_idx
                            //     .iter()
                            //     .map(|&grad_ij| grad_ij * grad_ij)
                            //     .sum()
                            //     .sqrt();
                            let norm_grad_j = grad_idx
                                .fold(F::zero(), |sum, &grad_ij| sum + grad_ij.powi(2))
                                .sqrt();
                            F::max(F::zero(), norm_grad_j - self.alpha)
                        }
                    }
                })
                .collect(),
        );
        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
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
        norm_rows
            .iter()
            .map(|&nrm_j| match nrm_j < self.alpha * self.gamma {
                true => self.alpha * nrm_j - nrm_j.powi(2) / (cast2 * self.gamma),
                false => self.gamma * self.alpha.powi(2) / cast2,
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
            Array1::<F>::zeros(value.len())
        } else if norm_value > g * tau {
            value.to_owned()
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
        let subdiff_dist = Array1::from_vec(
            grad.axis_iter(Axis(0))
                .zip(ws)
                .map(|(grad_idx, &j)| {
                    let W_j = W.slice(s![j, ..]);
                    let norm_W_j = W_j.map(|wj| wj.powi(2)).sum().sqrt();

                    if !W_j.iter().any(|&wij| wij == F::zero()) {
                        // distance of -grad_j to alpha * unit ball
                        // let norm_grad_j =
                        //     grad_idx.iter().map(|&grad_ij| grad_ij.powi(2)).sum().sqrt();
                        let norm_grad_j = grad_idx
                            .fold(F::zero(), |sum, &grad_ij| sum + grad_ij.powi(2))
                            .sqrt();
                        F::max(F::zero(), norm_grad_j - self.alpha)
                    } else if norm_W_j < self.alpha * self.gamma {
                        // distance of -grad_j to alpha * W[j] / ||W_j|| - W[j] / gamma
                        let scale = self.alpha / norm_W_j - F::one() / self.gamma;
                        // W_j.iter().map(|&W_ij| (W_ij * scale).powi(2)).sum().sqrt();
                        W_j.fold(F::zero(), |sum, &W_ij| sum + (W_ij * scale).powi(2))
                            .sqrt()
                    } else {
                        // distance of -grad to 0
                        // grad_idx.iter().map(|&grad_ij| grad_ij.powi(2)).sum().sqrt();
                        grad_idx
                            .fold(F::zero(), |sum, &grad_ij| sum + grad_ij.powi(2))
                            .sqrt()
                    }
                })
                .collect(),
        );
        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}
