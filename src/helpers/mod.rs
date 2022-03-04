#[cfg(test)]
mod tests;

pub mod prox {
    use crate::Float;
    use ndarray::{Array1, ArrayView1};

    pub fn soft_thresholding<F: Float>(x: F, threshold: F) -> F {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            F::zero()
        }
    }

    pub fn block_soft_thresholding<F: Float>(x: ArrayView1<F>, threshold: F) -> Array1<F> {
        let norm_x = x.dot(&x).sqrt();
        if norm_x < threshold {
            return Array1::<F>::zeros(x.len());
        }
        let scale = F::one() - threshold / norm_x;
        &x * scale
    }

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

pub mod helpers {
    use crate::datasets::csc_array::CSCArray;
    use crate::Float;
    use ndarray::Data;
    use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Axis, Ix1};
    use std::cmp::Ordering;

    pub fn compute_alpha_max<F: 'static + Float>(X: ArrayView2<F>, y: ArrayView1<F>) -> F {
        let n_samples = F::cast(X.shape()[0]);
        let Xty = X.t().dot(&y);
        let alpha_max = Xty.fold(F::zero(), |max_val, &x| x.abs().max(max_val));
        alpha_max / n_samples
    }

    pub fn compute_alpha_max_mtl<F: 'static + Float>(X: ArrayView2<F>, Y: ArrayView2<F>) -> F {
        let n_samples = X.shape()[0];
        let XtY = X.t().dot(&Y);
        let norms_XtY = XtY.map_axis(Axis(1), |xtyj| (xtyj.dot(&xtyj).sqrt()));
        let alpha_max = norms_XtY.fold(F::zero(), |max_val, &xty| max_val.max(xty.abs()));
        alpha_max / F::cast(n_samples)
    }

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

    pub fn argsort_by<S, F>(arr: &ArrayBase<S, Ix1>, mut compare: F) -> Vec<usize>
    where
        S: Data,
        F: FnMut(&S::Elem, &S::Elem) -> Ordering,
    {
        // https://github.com/rust-ndarray/ndarray/issues/1145
        let mut indices: Vec<usize> = (0..arr.len()).collect();
        indices.sort_unstable_by(move |&i, &j| compare(&arr[i], &arr[j]));
        indices
    }
}

pub mod cholesky {
    use crate::Float;
    use ndarray::{s, Array1, Array2};
    use thiserror::Error;

    #[derive(Debug, Clone, Error)]
    pub enum LinAlgError {
        #[error("Ill-conditioned matrix. Impossible to factorize.")]
        IllConditionedMatrix,
    }

    pub fn forward_substitution<F: Float>(mat: &Array2<F>) -> Result<Array1<F>, LinAlgError> {
        // Solves the linear system Lx = 1 with L upper triangular
        let mut res = Array1::<F>::zeros(mat.len());

        for i in 0..mat.len() {
            res[i] = (F::one() - res.dot(&mat.slice(s![i, ..]))) / mat[[i, i]];
        }

        Ok(res)
    }

    pub fn cholesky_factorization<F: Float>(mat: &Array2<F>) -> Result<Array2<F>, LinAlgError> {
        // O(n^3) time | O(n^2) space
        let n = mat.shape()[0];
        let m = mat.shape()[1];
        assert_eq!(
            n, m,
            "Cholesky-factored matrix must be a squared symmetric positive-definite matrix."
        );

        let mut L = Array2::<F>::zeros((n, n));

        for i in 0..n {
            for j in 0..(i + 1) {
                let mut sum = F::zero();
                for k in 0..j {
                    sum += L[[i, k]] * L[[j, k]];
                }

                match i == j {
                    true => {
                        if mat[[i, i]] < sum {
                            return Err(LinAlgError::IllConditionedMatrix);
                        }
                        L[[i, j]] = (mat[[i, i]] - sum).sqrt();
                    }
                    false => {
                        if L[[j, j]] < F::cast(1e-15) {
                            return Err(LinAlgError::IllConditionedMatrix);
                        }
                        L[[i, j]] = F::one() / L[[j, j]] * (mat[[i, j]] - sum);
                    }
                }
            }
        }

        Ok(L)
    }
}

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
