extern create ndarray;

use crate::estimators::*;

use ndarray::{Array1, Array2};

pub mod helpers {

    pub fn fill_random_vector(capacity: usize) -> Vec<f64> {
        let mut data_x: Vec<f64> = Vec::with_capacity(capacity);
        for _ in 0..data_x.capacity() {
            data_x.push(random());
        }
        data_x
    }

    pub fn generate_random_data(n_samples: usize, n_features: usize) -> (Array2<T>, Array1<T>) {
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

#[test]
fn test_kkt_check() {
    let (X, y) = helpers::generate_random_data(10, 30);
    let alpha_max = helpers::compute_alpha_max(X.view(), y.view());
    let alpha = alpha_max * 0.1;

    let clf = Lasso::new(alpha);
    let w = clf.fit(X.view(), y.view());

    let r = y - X.dot(&w);
    let xr = X.t().dot(&r);

    assert_delta_arr!(xr, Array1::zeros(30), alpha + 1e-12);
}

#[test]
fn test_null_weight() {
    let (X, y) = helpers::generate_random_data(10, 30);
    let alpha_max = helpers::compute_alpha_max(X.view(), y.view());

    let clf = Lasso::new(alpha_max);
    let w = clf.fit(X.view(), y.view());

    assert_delta_arr!(w, Array1::zeros(w.len()), 1e-9);
}

#[test]
fn test_sklearn() {

}
