extern crate ndarray;

use ndarray::{Array1, ArrayBase, ArrayView1, Ix1, OwnedRepr};

use super::Float;
use crate::helpers::prox::{prox_05, soft_thresholding};

#[cfg(test)]
mod tests;

pub trait Penalty<F>
where
    F: Float,
{
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

pub struct L1<F>
where
    F: Float,
{
    alpha: F,
}

impl<F> L1<F>
where
    F: Float,
{
    // Constructor
    pub fn new(alpha: F) -> Self {
        L1 { alpha }
    }
}

impl<F> Penalty<F> for L1<F>
where
    F: Float,
{
    /// Gets the current value of the penalty
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha * w.map(|x| (*x).abs()).sum()
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

/// MCP penalty
///

pub struct MCP<F: Float> {
    alpha: F,
    gamma: F,
}

impl<F: Float> MCP<F> {
    /// Constructor
    ///
    pub fn new(alpha: F, gamma: F) -> Self {
        MCP { alpha, gamma }
    }
}

impl<F: Float> Penalty<F> for MCP<F> {
    /// Gets the current value of the penalty
    fn value(&self, w: ArrayView1<F>) -> F {
        // With x >= 0
        // pen(x) = alpha * x - x^2 / (2 * gamma) if x =< gamma * alpha
        //          gamma * alpha 2 / 2           if x > gamma * alpha
        // value = sum_{j=1}^{n_features} pen(|w_j|)
        let mut s0 = Array1::from_elem(w.len(), false);
        for (idx, &wj) in w.iter().enumerate() {
            if wj.abs() < self.gamma * self.alpha {
                s0[idx] = true;
            }
        }
        let mut value = Array1::from_elem(w.len(), self.gamma * self.alpha.powi(2) / F::cast(2));
        for idx in 0..w.len() {
            value[idx] = self.alpha * w[idx].abs() - w[idx].powi(2) / (F::cast(2) * self.gamma);
        }
        value.fold(F::zero(), |sum, &valuej| sum + valuej)
    }

    /// Proximal operator
    fn prox_op(&self, value: F, stepsize: F) -> F {
        let tau = self.alpha * stepsize;
        let g = self.gamma / stepsize;
        if value.abs() <= tau {
            return F::zero();
        }
        if value.abs() > g * tau {
            return value;
        }
        return value.signum() * (value.abs() - tau) / (F::one() - F::one() / g);
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
                // Distance of -grad to alpha * [-1, 1]
                subdiff_dist[idx] = F::max(F::zero(), grad[idx].abs() - self.alpha)
            } else if w[j].abs() < self.alpha * self.gamma {
                // Distance of -grad_j to (alpha - abs(w[j])/gamma) * sign(w[j])
                subdiff_dist[idx] =
                    (grad[idx] + self.alpha * w[j].signum() - w[j] / self.gamma).abs();
            } else {
                // Distance of grad to zero
                subdiff_dist[idx] = grad[idx].abs();
            }

            if subdiff_dist[idx] > max_subdiff_dist {
                max_subdiff_dist = subdiff_dist[idx];
            }
        }
        (subdiff_dist, max_subdiff_dist)
    }
}

/// L05 penalty
///

pub struct L05<F: Float> {
    alpha: F,
}

impl<F: Float> L05<F> {
    /// Constructor
    ///
    pub fn new(alpha: F) -> Self {
        L05 { alpha }
    }
}

impl<F: Float> Penalty<F> for L05<F> {
    /// Gets the current value of the penalty
    fn value(&self, w: ArrayView1<F>) -> F {
        self.alpha * w.fold(F::zero(), |sum, &wj| sum + wj.abs().sqrt())
    }

    /// Proximal operator
    fn prox_op(&self, value: F, stepsize: F) -> F {
        prox_05(value, stepsize * self.alpha)
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
                subdiff_dist[idx] = F::zero();
            } else {
                subdiff_dist[idx] = (-grad[idx]
                    - w[j].signum() * self.alpha / (F::cast(2.) * w[j].abs().sqrt()))
                .abs();
            }

            if subdiff_dist[idx] > max_subdiff_dist {
                max_subdiff_dist = subdiff_dist[idx];
            }
        }
        (subdiff_dist, max_subdiff_dist)
    }
}
