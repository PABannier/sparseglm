extern crate ndarray;
extern crate num;
extern crate rand;
extern crate rand_distr;

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

pub mod helpers {
    use ndarray::{ArrayView1, ArrayView2};
    use num::Float;

    pub fn get_max_arr<T: Float>(arr: ArrayView1<T>) -> T {
        let mut max_val = T::neg_infinity();
        for j in 0..arr.len() {
            if arr[j] > max_val {
                max_val = arr[j];
            }
        }
        max_val
    }

    pub fn compute_alpha_max<T: 'static + Float>(X: ArrayView2<T>, y: ArrayView1<T>) -> T {
        let n_samples = T::from(X.shape()[0]).unwrap();
        let Xty = X.t().dot(&y);
        let Xty = Xty.map(|x| x.abs());
        let alpha_max = get_max_arr(Xty.view());
        alpha_max / n_samples
    }
}
