extern crate ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2};

use crate::datafits_multitask::*;
use crate::datasets::*;
use crate::helpers::test_helpers::*;
use crate::penalties_multitask::*;
use crate::solver_multitask::*;

#[test]
fn test_bcd_epoch() {
    let X = Array2::from_shape_vec((2, 3), vec![3.4, -1.2, 2.3, 9.8, -2.7, -0.2]).unwrap();
    let Y = Array2::from_shape_vec((2, 2), vec![1.2, -0.9, 2.3, -1.2]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let mut W = Array2::from_shape_vec((3, 2), vec![1.3, -1.4, 1.5, 1.3, -1.4, 1.5]).unwrap();
    let mut XW = Array2::<f64>::zeros((2, 2));
    general_mat_mul(1., &X, &W, 1., &mut XW);

    let dataset = DatasetBase::from((X, Y));

    let mut datafit = QuadraticMultiTask::default();
    datafit.initialize(&dataset);
    let penalty = L21::new(0.3);

    let solver = MultiTaskSolver {};

    solver.bcd_epoch(&dataset, &datafit, &penalty, &mut W, &mut XW, ws.view());

    let true_W = Array2::from_shape_vec(
        (3, 2),
        vec![
            0.74391824, 0.14846259, 1.32198528, 1.3466079, 0.05736997, 0.01897019,
        ],
    )
    .unwrap();
    let true_XW = Array2::from_shape_vec(
        (2, 2),
        vec![1.07489061, -1.06752524, 3.70956451, -2.18470201],
    )
    .unwrap();

    assert_array2d_all_close(W.view(), true_W.view(), 1e-8);
    assert_array2d_all_close(XW.view(), true_XW.view(), 1e-8);
}

#[test]
fn test_bcd_epoch_sparse() {
    let indptr = Array1::from_shape_vec(4, vec![0, 2, 3, 6]).unwrap();
    let indices = Array1::from_shape_vec(6, vec![0, 2, 2, 0, 1, 2]).unwrap();
    let data = Array1::from_shape_vec(6, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let X = CSCArray::new(data.view(), indices.view(), indptr.view());
    let Y = Array2::from_shape_vec((3, 2), vec![1.2, -0.9, 0.1, 1.2, -1., 3.2]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let mut W = Array2::from_shape_vec((3, 2), vec![2.1, -0.9, 3.4, 2.1, -0.9, 3.4]).unwrap();
    let mut XW = Array2::from_shape_vec((3, 2), vec![-1.5, 12.7, -4.5, 17., 9., 24.9]).unwrap();

    let dataset = DatasetBase::from((X, Y));

    let mut datafit = QuadraticMultiTask::default();
    datafit.initialize(&dataset);

    let penalty = L21::new(0.7);

    let solver = MultiTaskSolver {};

    solver.bcd_epoch(&dataset, &datafit, &penalty, &mut W, &mut XW, ws.view());

    let true_W = Array2::from_shape_vec(
        (3, 2),
        vec![
            -1.31384227,
            -11.88254406,
            2.17205356,
            2.02907842,
            -0.24093139,
            2.24817207,
        ],
    )
    .unwrap();
    let true_XW = Array2::from_shape_vec(
        (3, 2),
        vec![
            -2.27756783,
            -2.88985576,
            -1.20465695,
            11.24086037,
            2.44288781,
            -4.18882042,
        ],
    )
    .unwrap();

    assert_array2d_all_close(W.view(), true_W.view(), 1e-8);
    assert_array2d_all_close(XW.view(), true_XW.view(), 1e-8);
}
