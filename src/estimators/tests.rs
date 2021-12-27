extern crate ndarray;
extern crate rand;

use crate::estimators::*;

macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) {
            panic!();
        }
    };
}

macro_rules! assert_delta_arr {
    ($x:expr, $y: expr, $d: expr) => {
        assert_eq!($x.len(), $y.len());
        for i in 0..$x.len() {
            assert_delta!($x[i], $y[i], $d);
        }
    };
}

pub mod helpers {
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use num::Float;
    use rand::random;

    pub fn fill_random_vector(capacity: usize) -> Vec<f64> {
        let mut data_x: Vec<f64> = Vec::with_capacity(capacity);
        for _ in 0..data_x.capacity() {
            data_x.push(random());
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

#[test]
fn test_kkt_check() {
    let (X, y) = helpers::generate_random_data(10, 30);
    let alpha_max = helpers::compute_alpha_max(X.view(), y.view());
    let alpha = alpha_max * 0.1;

    let mut clf = Lasso::new(alpha);
    let w = clf.fit(X.view(), y.view());

    let r = y - X.dot(&w);
    let xr = X.t().dot(&r);

    assert_delta_arr!(xr, Array1::<f64>::zeros(30), alpha + 1e-12);
}

#[test]
fn test_null_weight() {
    let (X, y) = helpers::generate_random_data(10, 30);
    let alpha_max = helpers::compute_alpha_max(X.view(), y.view());

    let mut clf = Lasso::new(alpha_max);
    let w = clf.fit(X.view(), y.view());

    assert_delta_arr!(w, Array1::<f64>::zeros(w.len()), 1e-9);
}

#[test]
fn test_sklearn() {}
