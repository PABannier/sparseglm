extern crate ndarray;
extern crate ndarray_stats;
extern crate rand;
extern crate rand_distr;


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
        let norm_x = x.map(|&i| i * i).sum().sqrt();
        let mut prox_val = Array1::<F>::zeros(x.len());
        if norm_x < threshold {
            return prox_val;
        }
        let scale = F::one() - threshold / norm_x;
        for i in 0..x.len() {
            prox_val[i] = x[i] * scale;
        }
        prox_val
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
    use ndarray::{Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Ix1};
    use ndarray_stats::QuantileExt;
    use std::cmp::Ordering;

    pub fn compute_alpha_max<T>(X: ArrayView2<T>, y: ArrayView1<T>) -> T
    where
        T: 'static + Float,
    {
        let n_samples = T::from(X.shape()[0]).unwrap();
        let Xty = X.t().dot(&y);
        let Xty = Xty.map(|x| x.abs());
        let alpha_max = Xty.max().unwrap();
        *alpha_max / n_samples
    }

    pub fn compute_alpha_max_mtl<T>(X: ArrayView2<T>, Y: ArrayView2<T>) -> T
    where
        T: 'static + Float,
    {
        let n_features = X.shape()[1];
        let n_tasks = Y.shape()[1];
        let n_samples = X.shape()[0];
        let mut XtY = Array2::<T>::zeros((n_features, n_tasks));

        for i in 0..n_features {
            for j in 0..n_tasks {
                for l in 0..n_samples {
                    XtY[[i, j]] = XtY[[i, j]] + X[[l, i]] * Y[[l, j]];
                }
            }
        }

        let norms_XtY = XtY.map_axis(Axis(1), |xtyj| (xtyj.dot(&xtyj).sqrt()));
        let norms_XtY = norms_XtY.map(|xty| xty.abs());
        let alpha_max = norms_XtY.max().unwrap();
        *alpha_max / T::from(n_samples).unwrap()
    }

    pub fn compute_alpha_max_sparse<T>(X: &CSCArray<T>, y: ArrayView1<T>) -> T
    where
        T: 'static + Float,
    {
        let n_samples = T::from(y.len()).unwrap();
        let n_features = X.indptr.len() - 1;
        let mut Xty = Array1::<T>::zeros(n_features);
        for j in 0..n_features {
            for idx in X.indptr[j]..X.indptr[j + 1] {
                Xty[j] = Xty[j] + X.data[idx as usize] * y[X.indices[idx as usize] as usize];
            }
        }
        let Xty = Xty.map(|x| x.abs());
        let alpha_max = Xty.max().unwrap();
        *alpha_max / n_samples
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

    pub fn solve_lin_sys<T: 'static + Float>(
        A: ArrayView2<T>,
        b: ArrayView1<T>,
    ) -> Result<Array1<T>, &'static str> {
        // Concatenation
        let size = b.len();
        let mut system = Array2::<T>::zeros((size, size + 1));
        for i in 0..size {
            for j in 0..(size + 1) {
                system[[i, j]] = if j == size { b[i] } else { A[[i, j]] };
            }
        }

        // Echelon form
        for i in 0..size - 1 {
            for j in i..size - 1 {
                if system[[i, i]] == T::zero() {
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
            if system[[i, i]] == T::zero() {
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

        let mut x = Array1::<T>::zeros(size);
        for i in 0..size {
            if system[[i, i]] == T::zero() {
                return Err("Infinitely many solutions or singular matrix");
            } else {
                system[[i, size]] = system[[i, size]] / system[[i, i]];
                system[[i, i]] = T::one();
                x[i] = system[[i, size]];
            }
        }

        Ok(x)
    }
}

pub mod test_helpers {
    use crate::Float;
    use approx::AbsDiffEq;
    use ndarray::prelude::*;
    use ndarray::{linalg::general_mat_mul, Array1, Array2, ArrayView1, ArrayView2};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    pub fn assert_array_all_close<T>(x: ArrayView1<T>, y: ArrayView1<T>, delta: T)
    where
        T: Float + AbsDiffEq<Epsilon = T>,
    {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            if x[i].abs_diff_ne(&y[i], delta) {
                panic!("x: {}, y: {} ; with precision level {}", x[i], y[i], delta);
            }
        }
    }

    pub fn assert_array2d_all_close<T>(x: ArrayView2<T>, y: ArrayView2<T>, delta: T)
    where
        T: Float + AbsDiffEq<Epsilon = T>,
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
        let mut Y = Array2::<f64>::zeros((n_samples, n_tasks));
        general_mat_mul(1., &X, &true_W, 1., &mut Y);
        Y = Y + noise;

        (X, Y)
    }
}