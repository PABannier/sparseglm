extern crate ndarray;
extern crate rand;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2};

use crate::datasets::*;
use crate::estimators::*;
use crate::helpers::helpers::*;
use crate::helpers::test_helpers::*;

macro_rules! kkt_check_tests {
    ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (n_samples, n_features) = $value;
                let (X, y) = generate_random_data(n_samples, n_features);
                let dataset = DenseDatasetView::from((X.view(), y.view()));

                let alpha_max = compute_alpha_max(X.view(), y.view());
                let alpha = alpha_max * 0.5;

                let mut clf = Lasso::new(alpha, None);
                let w = clf.fit(&dataset);

                let r = y - X.dot(&w);
                let xr = X.t().dot(&r) / (n_samples as f64);

                assert_array_all_close(
                    xr.view(),
                    Array1::<f64>::zeros(n_features).view(),
                    alpha + 1e-8);
            }
        )*
    }
}

macro_rules! kkt_check_mtl_tests {
    ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (n_samples, n_features, n_tasks) = $value;
                let (X, Y) = generate_random_data_mtl(n_samples, n_features, n_tasks);
                let dataset = DenseDatasetView::from((X.view(), Y.view()));

                let alpha_max = compute_alpha_max_mtl(X.view(), Y.view());
                let alpha = alpha_max * 0.5;

                let mut clf = MultiTaskLasso::new(alpha, None);
                let W = clf.fit(&dataset);

                let mut XW = Array2::<f64>::zeros((n_samples, n_tasks));
                general_mat_mul(1., &X, &W, 1., &mut XW);

                let R = Y - XW;

                let mut XR = Array2::<f64>::zeros((n_features, n_tasks));
                general_mat_mul(1., &X.t(), &R, 1., &mut XR);

                assert_array2d_all_close(
                    XR.view(),
                    Array2::<f64>::zeros((n_features, n_tasks)).view(),
                    (n_samples as f64) * (alpha + 1e-8)
                );
            }
        )*
    }
}

kkt_check_tests! {
    kkt_check_small: (10, 30),
    kkt_check_medium: (100, 300),
    kkt_check_large: (500, 1000),
}

kkt_check_mtl_tests! {
    kkt_check_mtl_small: (10, 30, 8),
    kkt_check_mtl_medium: (70, 150, 20),
}

#[test]
fn test_null_weight() {
    let n_samples = 10;
    let n_features = 30;
    let (X, y) = generate_random_data(n_samples, n_features);
    let dataset = DenseDatasetView::from((X.view(), y.view()));
    let alpha_max = compute_alpha_max(X.view(), y.view());

    let mut clf = Lasso::new(alpha_max, None);
    let w = clf.fit(&dataset);

    assert_array_all_close(w.view(), Array1::<f64>::zeros(w.len()).view(), 1e-9);
}
