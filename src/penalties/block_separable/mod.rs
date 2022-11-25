use ndarray::{s, Array1, ArrayView1, ArrayView2, Axis};

use super::Float;
use crate::helpers::prox::block_soft_thresholding;

#[cfg(test)]
mod tests;

/// This trait provides three methods needed to update the weights in a multi-task
/// setting during the optimization routine.
pub trait MultiTaskPenalty<F: Float> {
    /// This method is called when evaluating the objective value.
    ///
    /// It is jointly used with ['DatafitMultiTask::value`] in order to compute the value
    /// of the objective.
    fn value(&self, W: ArrayView2<F>) -> F;

    /// This method computes the proximal gradient step during the update of the weights.
    /// For a given penalty, it implements its proximal operator.
    fn prox(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F>;

    /// This method is used when ranking the features to build the working set.
    /// It allows to compute the distance between the gradient of the datafit
    /// to the subdifferential of the penalty.
    ///
    /// It outputs the distances of the gradient of each feature to the subdifferential
    /// of the penalty, as well as the maximum distance.
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F);
}

/// L21 penalty
///
/// The multi-task counterpart of the [`Penalty::L1`] penalty. It is used in the Multi-Task
/// LASSO model and yields sparse solutions.
#[derive(Debug, Clone, PartialEq)]
pub struct L21<F: Float> {
    alpha: F,
}

impl<F: Float> L21<F> {
    // Instantiates a L21 penalty with a positive regularization hyperparameter.
    pub fn new(alpha: F) -> Self {
        L21 { alpha }
    }
}

impl<F: 'static + Float> MultiTaskPenalty<F> for L21<F> {
    /// Computes the L21-norm of the weights
    ///
    /// pen(X) = sum_{t=1}^T ||X_:,t||_2
    fn value(&self, W: ArrayView2<F>) -> F {
        self.alpha * W.map_axis(Axis(1), |Wj| (Wj.dot(&Wj).sqrt())).sum()
    }

    /// Applies the block soft-thresholding operator to a weight vector
    fn prox(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F> {
        block_soft_thresholding(value, self.alpha * stepsize)
    }

    /// Computes the distance of the gradient to the subdifferential
    ///
    /// The distance of the gradient to the subdifferential of L21 is:
    /// dist(grad, subdiff) = max(0, ||grad|| - alpha)             if ||W[j]|| = 0
    ///                       || grad + alpha * W[j] / ||W[j]|| || otherwise
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist =
            Array1::from_iter(grad.axis_iter(Axis(0)).zip(ws).map(|(grad_idx, &j)| {
                let W_j = W.slice(s![j, ..]);

                match W_j.iter().any(|&w_ij| w_ij != F::zero()) {
                    true => {
                        let norm_W_j = W_j.dot(&W_j).sqrt();
                        grad_idx
                            .iter()
                            .zip(W_j)
                            .fold(F::zero(), |sum, (&grad_ij, &w_ij)| {
                                sum + (grad_ij + self.alpha * w_ij / norm_W_j).powi(2)
                            })
                            .sqrt()
                    }
                    false => {
                        let norm_grad_j = grad_idx.dot(&grad_idx).sqrt();
                        F::max(F::zero(), norm_grad_j - self.alpha)
                    }
                }
            }));

        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}

/// The Block L1 + L2 penalty
///
/// A block convex penalty used by the Elastic Net model. It is used in a
/// multi-task regression setting and is a combination of a L2-regularized OLS
/// model (MultiTask Ridge) and a L21-regularized OLS (MultiTask Lasso).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockL1PlusL2<F: Float> {
    alpha: F,
    l21_ratio: F,
}

impl<F: Float> BlockL1PlusL2<F> {
    /// Instantiates a block L1 + L2 penalty with a positive regularization hyperparameter
    /// and a weighting hyperparameter between 0 and 1 that weights the amount of
    /// L21 and L2,2 regularizations.
    pub fn new(alpha: F, l21_ratio: F) -> Self {
        BlockL1PlusL2 { alpha, l21_ratio }
    }
}

impl<F: Float> MultiTaskPenalty<F> for BlockL1PlusL2<F> {
    /// Computes the block L1 + L2 - norm of the weights
    ///
    /// pen(X) = alpha * l21_ratio * ||X||_2,1 + alpha * (1 - l21_ratio) * ||X||_2,2^2 / 2
    fn value(&self, W: ArrayView2<F>) -> F {
        let norm_W_j = W.map_axis(Axis(1), |Wj| Wj.dot(&Wj).sqrt());
        self.alpha
            * (self.l21_ratio * norm_W_j.sum()
                + (F::one() - self.l21_ratio) * F::cast(0.5) * norm_W_j.dot(&norm_W_j))
    }

    /// Computes the proximal operator the Block L1 + L2 penalty for a weight vector
    fn prox(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F> {
        let prox = block_soft_thresholding(value, self.alpha * stepsize * self.l21_ratio);
        prox / (F::one() + stepsize * (F::one() - self.l21_ratio) * self.alpha)
    }

    /// Computes the distance of the gradient to the subdifferential
    ///
    /// The distance of the gradient to the subdifferential of Block L1 + L2 is:
    /// dist(grad, subdiff) = max(0, ||grad|| - alpha * l21_ratio)              if ||W[j]|| = 0
    ///                       || - grad - alpha * (W[j] / ||W[j]|| * l21_ratio
    ///                          + (1 - l21_ratio) * W[j]) ||                   otherwise
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist =
            Array1::from_iter(grad.axis_iter(Axis(0)).zip(ws).map(|(grad_idx, &j)| {
                let W_j = W.slice(s![j, ..]);

                match W_j.iter().any(|&w_ij| w_ij != F::zero()) {
                    true => {
                        let norm_W_j = W_j.dot(&W_j).sqrt();
                        grad_idx
                            .iter()
                            .zip(W_j)
                            .fold(F::zero(), |sum, (&grad_ij, &w_ij)| {
                                sum + (grad_ij
                                    + self.alpha
                                        * (w_ij / norm_W_j * self.l21_ratio
                                            + (F::one() - self.l21_ratio) * w_ij))
                                    .powi(2)
                            })
                            .sqrt()
                    }
                    false => {
                        let norm_grad_j = grad_idx.dot(&grad_idx);
                        F::max(F::zero(), norm_grad_j - self.alpha * self.l21_ratio)
                    }
                }
            }));

        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}

/// The Block Minimax Concave Penalty
///
/// A block non-convex penalty that yields sparser solutions than the L21 penalty and mitigates
/// the intrinsic L21-penalty bias.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockMCP<F: Float> {
    alpha: F,
    gamma: F,
}

impl<F: Float> BlockMCP<F> {
    /// Instantiates the Block Minimax Concave Penalty (MCP) with a given positive regularization
    /// and a shaping hyperparameter
    pub fn new(alpha: F, gamma: F) -> Self {
        BlockMCP { alpha, gamma }
    }
}

impl<F: Float> MultiTaskPenalty<F> for BlockMCP<F> {
    /// Computes the Block MCP for the weight matrix
    ///
    /// With W_j the j-th row of W, compute
    /// pen(||W_j||) = alpha * ||W_j|| - ||W_j||^2 / (2 * gamma)
    ///                if ||W_j|| =< gamma * alpha
    ///              = gamma * alpha ** 2 / 2
    ///                if ||W_j|| > gamma * alpha
    /// value = sum_{j=1}^{n_features} pen(||W_j||)
    fn value(&self, W: ArrayView2<F>) -> F {
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

    /// Computes the proximal operator of block MCP for a weight vector
    ///
    /// prox(x, threshold) = [0.]                                   ||x|| < alpha * threshold
    ///                       x                                     ||x|| > alpha * gamma
    ///                       (1 - alpha * threshold / ||x||) * x   otherwise
    ///                       / (1 - threshold / gamma)
    fn prox(&self, value: ArrayView1<F>, stepsize: F) -> Array1<F> {
        let cast1 = F::cast(1.);
        let tau = self.alpha * stepsize;
        let g = self.gamma / stepsize;
        let norm_value = value.dot(&value).sqrt();
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

    /// Computes the distance of the gradient to the subdifferential of block MCP
    ///
    /// dist(grad, subdiff) = max(0, ||grad|| - alpha)                          if ||W[j]|| = 0
    ///                       ||grad + (alpha / ||W[j]|| - 1 / gamma) * W[j]||  if ||W[j]|| < alpha * gamma
    ///                       ||grad||                                          otherwise
    fn subdiff_distance(
        &self,
        W: ArrayView2<F>,
        grad: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist =
            Array1::from_iter(grad.axis_iter(Axis(0)).zip(ws).map(|(grad_idx, &j)| {
                let W_j = W.slice(s![j, ..]);
                let norm_W_j = W_j.dot(&W_j).sqrt();

                match W_j.iter().any(|&w_ij| w_ij != F::zero()) {
                    false => {
                        // distance of -grad_j to alpha * unit_ball
                        let norm_grad_j = grad_idx.dot(&grad_idx).sqrt();
                        F::max(F::zero(), norm_grad_j - self.alpha)
                    }
                    _ => {
                        if norm_W_j < self.alpha * self.gamma {
                            // distance of -grad_j to alpha * W[j] / ||W_j|| - W[j] / gamma
                            let scale = self.alpha / norm_W_j - F::one() / self.gamma;
                            W_j.map(|&W_ij| (W_ij * scale).powi(2)).sum().sqrt()
                        } else {
                            // distance of -grad_j to 0
                            grad_idx.dot(&grad_idx).sqrt()
                        }
                    }
                }
            }));

        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}
