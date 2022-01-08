extern crate ndarray;
extern crate num;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2};

use crate::datafits_multitask::*;
use crate::helpers::test_helpers::*;
use crate::sparse::*;

#[test]
fn test_initialization_quadratic_mtl() {
    let X = Array2::from_shape_vec((2, 3), vec![3.4, 2.1, 2.3, 3.4, -1.2, 0.2]).unwrap();
    let Y = Array2::from_shape_vec((2, 2), vec![-3.4, 2.1, -0.3, 2.3]).unwrap();
    let mut df = QuadraticMultiTask::default();
    df.initialize(X.view(), Y.view());
    let XtY =
        Array2::from_shape_vec((3, 2), vec![-12.58, 14.96, -6.78, 1.65, -7.88, 5.29]).unwrap();
    let lipschitz = Array1::from_shape_vec(3, vec![11.56, 2.925, 2.665]).unwrap();
    assert_array2d_all_close(XtY.view(), df.get_XtY().view(), 1e-8);
    assert_array_all_close(lipschitz.view(), df.get_lipschitz().view(), 1e-8);
}

#[test]
fn test_initialization_sparse_quadratic_mtl() {
    let indptr = Array1::from_shape_vec(4, vec![0, 2, 3, 6]).unwrap();
    let indices = Array1::from_shape_vec(6, vec![0, 2, 2, 0, 1, 2]).unwrap();
    let data = Array1::from_shape_vec(6, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let X_sparse = CSCArray::new(data.view(), indices.view(), indptr.view());
    let X = Array2::from_shape_vec((3, 3), vec![1., 0., 4., 0., 0., 5., 2., 3., 6.]).unwrap();
    let Y = Array2::from_shape_vec((3, 2), vec![1., 3., 2., 2., 4., 1.]).unwrap();

    let mut df_sparse = QuadraticMultiTask::default();
    df_sparse.initialize_sparse(&X_sparse, Y.view());

    let mut df = QuadraticMultiTask::default();
    df.initialize(X.view(), Y.view());

    assert_array2d_all_close(df.get_XtY().view(), df_sparse.get_XtY().view(), 1e-8);
    assert_array_all_close(
        df.get_lipschitz().view(),
        df_sparse.get_lipschitz().view(),
        1e-8,
    );
}

#[test]
fn test_value_quadratic() {
    let X = Array2::from_shape_vec((2, 3), vec![3.6, 1.1, 2.2, 3.4, -1.2, 0.2]).unwrap();
    let Y = Array2::from_shape_vec((2, 2), vec![-3.3, 2.7, 1.2, 4.5]).unwrap();
    let W = Array2::from_shape_vec((3, 2), vec![-3.2, -0.21, 2.3, -2.3, -1.2, 3.2]).unwrap();
    let mut XW = Array2::<f64>::zeros((2, 2));
    general_mat_mul(1., &X, &W, 1., &mut XW);
    let mut df = QuadraticMultiTask::default();
    df.initialize(X.view(), Y.view());
    let val = df.value(Y.view(), XW.view());
    assert_eq!(val, 75.299203);
}

#[test]
fn test_gradient_quadratic() {
    let X = Array2::from_shape_vec((2, 3), vec![3.0, 1.1, 3.2, 3.4, -1.2, 0.2]).unwrap();
    let Y = Array2::from_shape_vec((2, 2), vec![-3.3, 2.4, -1.2, 0.2]).unwrap();
    let W = Array2::from_shape_vec((3, 2), vec![-3.2, -0.25, 3.3, -1.2, -2.3, 1.2]).unwrap();
    let mut XW = Array2::<f64>::zeros((2, 2));
    general_mat_mul(1., &X, &W, 1., &mut XW);
    let mut df = QuadraticMultiTask::default();
    df.initialize(X.view(), Y.view());
    let grad = df.gradient_j(X.view(), XW.view(), 1);
    let true_grad = Array1::from_shape_vec(2, vec![2.9435, -0.7245]).unwrap();
    assert_array_all_close(grad.view(), true_grad.view(), 1e-6);
}

#[test]
fn test_gradient_sparse_quadratic() {
    let indptr = Array1::from_shape_vec(4, vec![0, 2, 3, 6]).unwrap();
    let indices = Array1::from_shape_vec(6, vec![0, 2, 2, 0, 1, 2]).unwrap();
    let data = Array1::from_shape_vec(6, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let X_sparse = CSCArray::new(data.view(), indices.view(), indptr.view());

    let X = Array2::from_shape_vec((3, 3), vec![1., 0., 4., 0., 0., 5., 2., 3., 6.]).unwrap();
    let Y = Array2::from_shape_vec((3, 2), vec![1., 3., 2., -2., 3., -1.]).unwrap();
    let W = Array2::from_shape_vec((3, 2), vec![-3.2, -0.25, 3.3, 0.1, 2.3, 1.]).unwrap();
    let mut XW = Array2::<f64>::zeros((3, 2));
    general_mat_mul(1., &X, &W, 1., &mut XW);

    let mut df = QuadraticMultiTask::default();
    let mut df_sparse = QuadraticMultiTask::default();
    df.initialize(X.view(), Y.view());
    df_sparse.initialize_sparse(&X_sparse, Y.view());

    let grad = df.gradient_j(X.view(), XW.view(), 1);
    let grad_sparse = df_sparse.gradient_j_sparse(&X_sparse, XW.view(), 1);
    assert_array_all_close(grad.view(), grad_sparse.view(), 1e-6);
}
