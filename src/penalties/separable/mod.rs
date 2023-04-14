use ndarray::{Array1, ArrayBase, ArrayView1, Ix1, OwnedRepr};

use super::Float;
use crate::utils::prox_funcs::{box_projection, prox_05, soft_thresholding};

#[cfg(test)]
mod tests;

/// This trait provides three methods needed to update the weights during the
/// optimization routine.
pub trait Penalty<F: Float> {
    /// This method is called when evaluating the objective value.
    ///
    /// It is jointly used with [`Datafit::value`] in order to compute the value
    /// of the objective.
    fn value(&self, w: ArrayView1<F>) -> F;

    /// This method computes the proximal gradient step during the update of the
    /// weights. For a given penalty, it implements its proximal operator.
    fn prox(&self, value: F, step_size: F) -> F;

    /// This method is used when ranking the features to build the working set.
    /// It allows to compute the distance between the gradient of the datafit
    /// to the subdifferential of the penalty.
    ///
    /// It outputs the distances of the gradient of each feature to the
    /// subdifferential of the penalty, as well as the maximum distance.
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F);
}

/// The L1 penalty
///
/// A widely-used penalty made popular by the LASSO model. It is used in a
/// regression setting and yields sparse solutions. Note that LASSO yields a
/// biased solution compared to the ordinary least square solution.
#[derive(Debug, Clone, PartialEq)]
pub struct L1<F: Float> {
    alpha: F,
}

impl<F: Float> L1<F> {
    /// Instantiates a L1 penalty with a positive regularization hyperparameter.
    pub fn new(alpha: F) -> Self {
        L1 { alpha }
    }
}

impl<F: Float> Penalty<F> for L1<F> {
    /// Computes the L1-norm of the weights
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha * w.iter().map(|&wj| wj.abs()).sum()
    }

    /// Applies the soft-thresholding operator to a weight scalar
    fn prox(&self, value: F, stepsize: F) -> F {
        soft_thresholding(value, self.alpha * stepsize)
    }

    /// Computes the distance of the gradient to the subdifferential
    ///
    /// The distance of the gradient to the subdifferential of L1 is:
    /// dist(grad, subdiff) = max(0, |grad| - alpha)         if w[j] = 0
    ///                       |- grad - sign(w[j]) * alpha|  otherwise
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist = Array1::from_vec(
            grad.iter()
                .zip(ws)
                .map(|(&grad_idx, &j)| match w[j] == F::zero() {
                    true => F::max(F::zero(), grad_idx.abs() - self.alpha),
                    false => (-grad_idx - w[j].signum() * self.alpha).abs(),
                })
                .collect(),
        );
        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}

/// The L1 + L2 penalty
///
/// A convex penalty used by the Elastic Net model. It is used in a regression setting
/// and is a combination of a L2-regularized OLS model (Ridge) and a L1-regularized OLS
/// model (Lasso).
#[derive(Debug, Clone, PartialEq)]
pub struct L1PlusL2<F: Float> {
    alpha: F,
    l1_ratio: F,
}

impl<F: Float> L1PlusL2<F> {
    /// Instantiates a L1 + L2 penalty with a positive regularization hyperparameter
    /// and a weighting hyperparameter between 0 and 1 that weights the amount of L1
    /// and L2 regularizations.
    pub fn new(alpha: F, l1_ratio: F) -> Self {
        L1PlusL2 { alpha, l1_ratio }
    }
}

impl<F: Float> Penalty<F> for L1PlusL2<F> {
    /// Computes the L1 + L2 - norm of the weights
    ///
    /// pen(x) = alpha * l1_ratio * ||x||_1 + alpha * (1 - l1_ratio) * ||x||_2^2 / 2
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha
            * (self.l1_ratio * w.iter().map(|&wj| wj.abs()).sum()
                + F::cast(0.5) * (F::one() - self.l1_ratio) * w.dot(&w))
    }

    /// Computes the proximal operator of the L1 + L2 penalty for a weight scalar
    fn prox(&self, value: F, stepsize: F) -> F {
        let prox = soft_thresholding(value, self.l1_ratio * self.alpha * stepsize);
        prox / (F::one() + stepsize * (F::one() - self.l1_ratio) * self.alpha)
    }

    /// Computes the distance of the gradient to the subdifferential
    ///
    /// The distance of the gradient to the subdifferential of L1 + L2 is:
    /// dist(grad, subdiff) = max(0, |grad| - alpha * l1_ratio)         if w[j] = 0
    ///                       |- grad - alpha * (sign(w[j]) * l1_ratio
    ///                        + (1 - l1_ratio) * w[j])|                otherwise
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        let subdiff_dist = Array1::from_vec(
            grad.iter()
                .zip(ws)
                .map(|(&grad_idx, &j)| match w[j] == F::zero() {
                    true => F::max(F::zero(), grad_idx.abs() - self.alpha * self.l1_ratio),
                    false => (-grad_idx
                        - self.alpha
                            * (w[j].signum() * self.l1_ratio + (F::one() - self.l1_ratio) * w[j]))
                        .abs(),
                })
                .collect(),
        );
        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}

/// The Indicator box penalty
///
/// A box-constraint penalty used in the dual formulation of the SVM prolem.
#[derive(Debug, Clone, PartialEq)]
pub struct IndicatorBox<F: Float> {
    alpha: F,
}

impl<F: Float> IndicatorBox<F> {
    /// Instantiates an Indicator Box penalty with a given positive
    /// regularization and a shaping hyperparameter
    pub fn new(alpha: F) -> Self {
        IndicatorBox { alpha }
    }
}

impl<F: Float> Penalty<F> for IndicatorBox<F> {
    fn value(&self, w: ArrayView1<F>) -> F {
        let max_val = w.fold(F::neg_infinity(), |max_val, &w_i| F::max(max_val, w_i));
        let min_val = w.fold(F::infinity(), |min_val, &w_i| F::min(min_val, w_i));
        // let mut max_val = -F::infinity();
        // let mut min_val = F::infinity();
        // w.for_each(|&w_i| {
        //     max_val = F::max(max_val, w_i);
        //     min_val = F::min(min_val, w_i);
        // });
        if max_val > self.alpha {
            F::infinity()
        } else if min_val < 0 {
            F::infinity()
        } else {
            F::zero()
        }
    }

    fn prox(&self, value: F, step_size: F) -> F {
        box_projection(value, 0, self.alpha)
    }

    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        Array1::from_vec(grad.iter().zip(ws).map(|(&grad_idx, &j)| match w[j] {
            0 => F::max(F::zero(), -grad_idx),
            self::alpha => F::max(F::zero(), grad_idx),
            _ => grad_idx.abs(),
        }))
    }
}

/// The Minimax concave penalty
///
/// A non-convex penalty that yields sparser solutions than the L1 penalty and
/// mitigates the intrinsic L1-penalty bias.
#[derive(Debug, Clone, PartialEq)]
pub struct MCP<F: Float> {
    alpha: F,
    gamma: F,
}

impl<F: Float> MCP<F> {
    /// Instantiates a Minimax Concave Penalty (MCP) with a given positive
    /// regularization and a shaping hyperparameter
    pub fn new(alpha: F, gamma: F) -> Self {
        MCP { alpha, gamma }
    }
}

impl<F: Float> Penalty<F> for MCP<F> {
    /// Computes the MCP for the weight vector
    ///
    /// With x >= 0
    /// pen(x) = alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
    ///          gamma * alpha 2 / 2           if x > gamma * alpha
    /// value = sum_{j=1}^{n_features} pen(|w_j|)
    fn value(&self, w: ArrayView1<F>) -> F {
        let cast2 = F::cast(2.);
        w.iter()
            .map(|&wj| match wj.abs() < self.gamma * self.alpha {
                true => self.alpha * wj.abs() - wj.powi(2) / (cast2 * self.gamma),
                false => self.gamma * self.alpha.powi(2) / cast2,
            })
            .sum()
    }

    /// Computes the proximal operator of MCP for a weight scalar
    ///
    /// prox(x, threshold) = 0.                                     if |x| < alpha * threshold
    ///                      x                                      if |x| > alpha * gamma
    ///                      sign(x) * (|x| - alpha * threshold)
    ///                      / (1 - threshold / gamma)              otherwise
    fn prox(&self, value: F, stepsize: F) -> F {
        let tau = self.alpha * stepsize;
        let g = self.gamma / stepsize;
        if value.abs() <= tau {
            F::zero()
        } else if value.abs() > g * tau {
            value
        } else {
            value.signum() * (value.abs() - tau) / (F::one() - F::one() / g)
        }
    }

    /// Computes the distance of the gradient to the subdifferential of MCP
    ///
    /// dist(grad, subdiff) = max(0, |grad| - alpha)                      if w[j] = 0
    ///                       |grad + alpha * sign(w[j]) - w[j] / gamma|  if |w[j]| < alpha * gamma
    ///                       |grad|                                      otherwise
    fn subdiff_distance(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F) {
        let subdiff_dist = Array1::from_vec(
            grad.iter()
                .zip(ws)
                .map(|(&grad_idx, &j)| {
                    if w[j] == F::zero() {
                        F::max(F::zero(), grad_idx.abs() - self.alpha)
                    } else if w[j].abs() < self.alpha * self.gamma {
                        (grad_idx + self.alpha * w[j].signum() - w[j] / self.gamma).abs()
                    } else {
                        grad_idx.abs()
                    }
                })
                .collect(),
        );
        let max_dist = subdiff_dist.fold(F::neg_infinity(), |max_val, &dist| F::max(max_val, dist));
        (subdiff_dist, max_dist)
    }
}

/// The L05 penalty
///
/// A non-convex penalty based on the quasi-norm l0.5. It creates sparser
/// solutions than the L1 penalty.
#[derive(Debug, Clone, PartialEq)]
pub struct L05<F: Float> {
    alpha: F,
}

impl<F: Float> L05<F> {
    /// Instantiates a L05 penalty with a given regularization hyperparameter
    pub fn new(alpha: F) -> Self {
        L05 { alpha }
    }
}

impl<F: Float> Penalty<F> for L05<F> {
    /// Computes the L05 penalty value for the weight vector
    ///
    /// pen(x) = alpha * ||x||_0.5
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha * w.map(|wj| wj.abs().sqrt()).sum()
    }

    /// Computes the proximal operator of the L0.5 norm for a weight scalar
    fn prox(&self, value: F, stepsize: F) -> F {
        prox_05(value, stepsize * self.alpha)
    }

    /// No distance to the subdifferential is computed for the L0.5 norm since
    /// the subdifferential is the real line, therefore the distance to the
    /// gradient is always 0. This makes this criterion uninformative to build
    /// the working sets. This penalty relies instead on the violation of the
    /// fixed point iterate schema.
    fn subdiff_distance(
        &self,
        _w: ArrayView1<F>,
        _grad: ArrayView1<F>,
        ws: ArrayView1<usize>,
    ) -> (Array1<F>, F) {
        (Array1::<F>::zeros(ws.len()), F::zero())
    }
}

// SCAD penalty
// pub struct SCAD<F: Float> {
//     alpha: F,
//     gamma: F,
// }

// impl<F: Float> SCAD<F> {
//     /// Constructor
//     ///
//     pub fn new(alpha: F, gamma: F) -> Self {
//         SCAD { alpha, gamma }
//     }
// }

// impl<F: Float> Penalty<F> for SCAD<F> {
//     /// Gets the current value of the penalty
//     fn value(&self, w: ArrayView1<F>) -> F {
//         // With x >= 0
//         // pen(x) = alpha * x                                                   if x =< alpha
//         //          (2 * gamma * alpha * x - x^2 - alpha^2) / (2 * (gamma - 1)) if gamma < x < gamma * lambda
//         //          alpha^2 * (gamma + 1) / 2                                   if x >= gamma * alpha
//         let mut s0 = Array1::from_elem(w.len(), false);
//         let mut s1 = Array1::from_elem(w.len(), false);
//         for (idx, &wj) in w.iter().enumerate() {
//             if wj.abs() <= self.alpha {
//                 s0[idx] = true;
//             }
//             if wj.abs() >= self.alpha * self.gamma {
//                 s1[idx] = true;
//             }
//         }
//         let mut value = Array1::from_elem(
//             w.len(),
//             self.alpha * self.alpha * (self.gamma + F::one()) * F::cast(0.5),
//         );
//         for idx in 0..w.len() {
//             if s0[idx] {
//                 value[idx] = self.alpha * w[idx].abs();
//             }
//             if !s0[idx] && !s1[idx] {
//                 value[idx] = (F::cast(2) * self.alpha * self.gamma * w[idx].abs()
//                     - w[idx].powi(2)
//                     - self.alpha.powi(2))
//                     / (F::cast(2) * (self.gamma - F::one()));
//             }
//         }
//         value.fold(F::zero(), |sum, &valuej| sum + valuej)
//     }

//     /// Proximal operator
//     fn prox(&self, value: F, stepsize: F) -> F {
//         let tau = self.alpha * stepsize;
//         let g = self.gamma / stepsize;
//         if value.abs() <= F::cast(2) * tau {
//             return soft_thresholding(value, tau);
//         } else if value.abs() > g * tau {
//             return value;
//         } else {
//             return (g - F::one()) / (g - F::cast(2))
//                 * soft_thresholding(value, g * tau / (g - F::one()));
//         }
//     }

//     /// Computes the distance of the gradient to the subdifferential
//     fn subdiff_distance(
//         &self,
//         w: ArrayView1<F>,
//         grad: ArrayView1<F>,
//         ws: ArrayView1<usize>,
//     ) -> (ArrayBase<OwnedRepr<F>, Ix1>, F) {
//         let ws_size = ws.len();
//         let mut subdiff_dist = Array1::<F>::zeros(ws_size);
//         let mut max_subdiff_dist = F::neg_infinity();
//         for (idx, &j) in ws.iter().enumerate() {
//             if subdiff_dist[idx] > max_subdiff_dist {
//                 max_subdiff_dist = subdiff_dist[idx];
//             }
//         }
//         (subdiff_dist, max_subdiff_dist)
//     }
// }
