#[cfg(test)]
mod tests;

/// This module implements the proximal operator of some penalties.
pub mod prox {
    use crate::Float;
    use ndarray::{Array1, ArrayView1};

    /// The soft-thresholding operator is the proximal operator used by
    /// [`Penalty::L1`].
    pub fn soft_thresholding<F: Float>(x: F, threshold: F) -> F {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            F::zero()
        }
    }

    /// The block soft-thresholding operator is the proximal operator used by
    /// [`MultiTaskPenalty::L21`].
    pub fn block_soft_thresholding<F: Float>(x: ArrayView1<F>, threshold: F) -> Array1<F> {
        let norm_x = x.map(|&xi| xi.powi(2)).sum().sqrt();
        if norm_x < threshold {
            return Array1::<F>::zeros(x.len());
        }
        let scale = F::one() - threshold / norm_x;
        &x * scale
    }

    /// This function implements the proximal operator of [`Penalty::L05`].
    pub fn prox_05<F: Float>(x: F, threshold: F) -> F {
        let t = F::cast(F::cast(3. / 2.) * threshold.powf(F::cast(2. / 3.)));
        if x.abs() < t {
            return F::zero();
        }
        x * F::cast(2. / 3.)
            * (F::one()
                + F::cos(
                    F::cast(2. / 3.)
                        * F::acos(
                            -F::cast(F::cast(3).powf(F::cast(3. / 2.)) / F::cast(4.))
                                * threshold
                                * x.abs().powf(F::cast(-3. / 2.)),
                        ),
                ))
    }
}

/// This module contains helper functions to compute the maximum regularization
/// hyperparameter for sparse models. A regularization hyperparameter value
/// larger than this maximum value yields a null solution.
pub mod helpers {
    use crate::datasets::csc_array::CSCArray;
    use crate::Float;
    use ndarray::Data;
    use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Ix1};
    use std::cmp::Ordering;

    /// This function computes the maximum regularization hyperparameter value
    /// for single task models.
    pub fn compute_alpha_max<F: 'static + Float>(X: ArrayView2<F>, y: ArrayView1<F>) -> F {
        let n_samples = F::cast(X.shape()[0]);
        let Xty = X.t().dot(&y);
        let alpha_max = Xty.fold(F::zero(), |max_val, &x| x.abs().max(max_val));
        alpha_max / n_samples
    }

    /// This function computes the maximum regularization hyperparameter value
    /// for multi task models.
    pub fn compute_alpha_max_mtl<F: 'static + Float>(X: ArrayView2<F>, Y: ArrayView2<F>) -> F {
        let n_samples = X.shape()[0];
        let XtY = X.t().dot(&Y);
        let norms_XtY = XtY.map_axis(Axis(1), |xtyj| (xtyj.dot(&xtyj).sqrt()));
        let alpha_max = norms_XtY.fold(F::zero(), |max_val, &xty| max_val.max(xty.abs()));
        alpha_max / F::cast(n_samples)
    }

    /// Much like [`compute_alpha_max`], this function computes the maximum
    /// regularization hyperparameter value for single task models with
    /// sparse design matrix.
    pub fn compute_alpha_max_sparse<F: 'static + Float>(X: &CSCArray<F>, y: ArrayView1<F>) -> F {
        let n_samples = F::cast(y.len());
        let n_features = X.indptr.len() - 1;
        let mut Xty = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            for idx in X.indptr[j]..X.indptr[j + 1] {
                Xty[j] = Xty[j] + X.data[idx as usize] * y[X.indices[idx as usize] as usize];
            }
        }
        let alpha_max = Xty.fold(F::zero(), |max_val, &x| max_val.max(x.abs()));
        alpha_max / n_samples
    }

    /// This is a helper method that sorts the indices of an array based on some
    /// `compare` closure. It is used in [`construct_ws_from_kkt`] in order to
    /// sort the features in the working set based on how far their gradient
    /// are from the penalty subdifferential.
    /// Reference: `https://github.com/rust-ndarray/ndarray/issues/1145`
    pub fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
    where
        S: Data,
        F: FnMut(&S::Elem, &S::Elem) -> Ordering,
    {
        let mut indices: Vec<usize> = (0..arr.len()).collect();
        indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
        indices
    }

    /// This function solves a linear system using Gaussian elimination. It is
    /// called in [`anderson_acceleration`] functions to invert the extrapolation
    /// matrix. We made the choice not to use a BLAS subroutine since it introduces
    /// unsafe code and a significantly larger bundle size.
    pub fn solve_lin_sys<F: 'static + Float>(
        A: ArrayView2<F>,
        b: ArrayView1<F>,
    ) -> Result<Array1<F>, &'static str> {
        // Concatenation
        let size = b.len();
        let mut system = Array2::<F>::zeros((size, size + 1));
        for i in 0..size {
            for j in 0..(size + 1) {
                system[[i, j]] = if j == size { b[i] } else { A[[i, j]] };
            }
        }

        // Echelon form
        for i in 0..size - 1 {
            for j in i..size - 1 {
                if system[[i, i]] == F::zero() {
                    continue;
                } else {
                    let factor = system[[j + 1, i]] / system[[i, i]];
                    for k in i..size + 1 {
                        system[[j + 1, k]] = system[[j + 1, k]] - factor * system[[i, k]];
                    }
                }
            }
        }

        // Gaussian eliminated
        for i in (1..size).rev() {
            if system[[i, i]] == F::zero() {
                continue;
            } else {
                for j in (1..i + 1).rev() {
                    let factor = system[[j - 1, i]] / system[[i, i]];
                    for k in (0..size + 1).rev() {
                        system[[j - 1, k]] = system[[j - 1, k]] - factor * system[[i, k]];
                    }
                }
            }
        }

        let mut x = Array1::<F>::zeros(size);
        for i in 0..size {
            if system[[i, i]] == F::zero() {
                return Err("Infinitely many solutions or singular matrix");
            } else {
                system[[i, size]] = system[[i, size]] / system[[i, i]];
                system[[i, i]] = F::one();
                x[i] = system[[i, size]];
            }
        }

        Ok(x)
    }
}

/// This module contains helpers functions to efficiently write tests.
pub mod test_helpers {
    use crate::Float;
    use approx::AbsDiffEq;
    use ndarray::prelude::*;
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    pub fn assert_array_all_close<F>(x: ArrayView1<F>, y: ArrayView1<F>, delta: F)
    where
        F: Float + AbsDiffEq<Epsilon = F>,
    {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            if x[i].abs_diff_ne(&y[i], delta) {
                panic!("x: {}, y: {} ; with precision level {}", x[i], y[i], delta);
            }
        }
    }

    pub fn assert_array2d_all_close<F>(x: ArrayView2<F>, y: ArrayView2<F>, delta: F)
    where
        F: Float + AbsDiffEq<Epsilon = F>,
    {
        assert_eq!(x.shape()[0], y.shape()[0]);
        assert_eq!(x.shape()[1], y.shape()[1]);
        for i in 0..x.shape()[0] {
            for j in 0..x.shape()[1] {
                if x[[i, j]].abs_diff_ne(&y[[i, j]], delta) {
                    panic!(
                        "x: {}, y: {} ; with precision level {}",
                        x[[i, j]],
                        y[[i, j]],
                        delta
                    );
                }
            }
        }
    }

    pub fn fill_random_vector(capacity: usize) -> Vec<f64> {
        let mut r = StdRng::seed_from_u64(42);
        let normal = Normal::new(0., 1.).unwrap();

        let mut data_x: Vec<f64> = Vec::with_capacity(capacity);
        for _ in 0..data_x.capacity() {
            data_x.push(normal.sample(&mut r));
        }
        data_x
    }

    pub fn generate_random_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
        let data_x = fill_random_vector(n_samples * n_features);
        let data_w = fill_random_vector(n_features);
        let data_e = fill_random_vector(n_samples);
        let X = Array2::from_shape_vec((n_samples, n_features).f(), data_x).unwrap();
        let true_w = Array1::from_shape_vec(n_features, data_w).unwrap();
        let noise = Array1::from_shape_vec(n_samples, data_e).unwrap();
        let y = X.dot(&true_w) + noise;

        (X, y)
    }

    pub fn generate_random_data_mtl(
        n_samples: usize,
        n_features: usize,
        n_tasks: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        let data_x = fill_random_vector(n_samples * n_features);
        let data_w = fill_random_vector(n_features * n_tasks);
        let data_e = fill_random_vector(n_samples * n_tasks);
        let X = Array2::from_shape_vec((n_samples, n_features).f(), data_x).unwrap();
        let true_W = Array2::from_shape_vec((n_features, n_tasks), data_w).unwrap();
        let noise = Array2::from_shape_vec((n_samples, n_tasks), data_e).unwrap();
        let Y = X.dot(&true_W) + noise;
        (X, Y)
    }
}
