extern crate ndarray;
extern crate ndarray_stats;
extern crate num;
extern crate rand;
extern crate rand_distr;

#[cfg(test)]
mod tests;

pub mod helpers {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use ndarray_stats::QuantileExt;
    use num::Float;

    use crate::sparse::*;

    pub fn compute_alpha_max<T: 'static + Float>(X: ArrayView2<T>, y: ArrayView1<T>) -> T {
        let n_samples = T::from(X.shape()[0]).unwrap();
        let Xty = X.t().dot(&y);
        let Xty = Xty.map(|x| x.abs());
        let alpha_max = Xty.max().unwrap();
        *alpha_max / n_samples
    }

    pub fn compute_alpha_max_sparse<T: 'static + Float + std::fmt::Debug>(
        X: &CSCArray<T>,
        y: ArrayView1<T>,
    ) -> T {
        let n_samples = T::from(y.len()).unwrap();
        let n_features = X.indptr.len() - 1;
        let mut Xty = Array1::<T>::zeros(n_features);
        for j in 0..n_features {
            for idx in X.indptr[j]..X.indptr[j + 1] {
                Xty[j] = Xty[j] + X.data[idx] * y[X.indices[idx]];
            }
        }
        let Xty = Xty.map(|x| x.abs());
        let alpha_max = Xty.max().unwrap();
        *alpha_max / n_samples
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
    use ndarray::{Array1, Array2, ArrayView1};
    use num::Float;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};
    use std::fmt::Display;

    pub fn assert_array_all_close<T: Float + Display>(
        x: ArrayView1<T>,
        y: ArrayView1<T>,
        delta: T,
    ) {
        assert_eq!(x.len(), y.len());
        for i in 0..x.len() {
            if !(T::abs(x[i] - y[i]) < delta) {
                panic!("x: {}, y: {} ; with precision level {}", x[i], y[i], delta);
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

        let X = Array2::from_shape_vec((n_samples, n_features), data_x).unwrap();
        let true_w = Array1::from_shape_vec(n_features, data_w).unwrap();
        let noise = Array1::from_shape_vec(n_samples, data_e).unwrap();
        let y = X.dot(&true_w) + noise;

        (X, y)
    }
}
