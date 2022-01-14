extern crate ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::{Array1, Array2};

use crate::bcd::*;
use crate::datafits::*;
use crate::datasets::csc_array::CSCArray;
use crate::helpers::test_helpers::*;
use crate::penalties::*;
use crate::solvers::*;

#[test]
fn test_kkt_violation() {
    let X = Array2::from_shape_vec((2, 3), vec![1.6, -1.3, 2.9, 10.8, -3.8, -0.1]).unwrap();
    let Y = Array2::from_shape_vec((2, 3), vec![1.4, -0.2, 0.2, 1.2, -3., -0.12]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let W = Array2::from_shape_vec((3, 3), vec![0.2, -0.3, 1.5, 0.2, -0.3, 1.5, 0.2, -0.3, 1.5])
        .unwrap();
    let mut XW = Array2::<f64>::zeros((2, 3));
    general_mat_mul(1., &X, &W, 1., &mut XW);

    let dataset = DatasetBase::from((X, Y));

    let mut datafit = QuadraticMultiTask::default();
    datafit.initialize(&dataset);
    let penalty = L21::new(0.3);

    let solver = Solver::new();

    let (kkt, kkt_max) =
        solver.kkt_violation(&dataset, W.view(), XW.view(), ws.view(), &datafit, &penalty);
    let true_kkt = Array1::from_shape_vec(3, vec![60.66759347, 22.63130826, 6.6374834]).unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-6);
    assert_eq!(kkt_max, 60.66759346931925);
}

#[test]
fn test_kkt_violation_sparse() {
    let data =
        Array1::from_shape_vec(4, vec![0.74272001, 0.95888754, 0.8886366, 0.19782359]).unwrap();
    let indices = Array1::from_shape_vec(4, vec![0, 1, 0, 0]).unwrap();
    let indptr = Array1::from_shape_vec(6, vec![0, 1, 2, 2, 3, 4]).unwrap();

    let X = CSCArray::new(data.view(), indices.view(), indptr.view());
    let Y = Array2::from_shape_vec((2, 3), vec![0.3, -2.3, 0.8, 1.2, -3.2, 0.1]).unwrap();
    let ws = Array1::from_shape_vec(5, (0..5).collect()).unwrap();

    let dataset = DatasetBase::from((X, Y));

    let W = Array2::from_shape_vec(
        (5, 3),
        vec![
            0.2, -0.3, 1.3, 3.4, -1.2, 0.2, -0.3, 1.3, 3.4, -1.2, 0.2, -0.3, 1.3, 3.4, -1.2,
        ],
    )
    .unwrap();
    let XW = Array2::from_shape_vec(
        (2, 3),
        vec![
            -0.66064925,
            0.62751152,
            0.46155673,
            3.26021764,
            -1.15066505,
            0.19177751,
        ],
    )
    .unwrap();

    let mut datafit = QuadraticMultiTask::default();
    datafit.initialize(&dataset);
    let penalty = L21::new(0.3);

    let solver = Solver::new();

    let (kkt, kkt_max) =
        solver.kkt_violation(&dataset, W.view(), XW.view(), ws.view(), &datafit, &penalty);
    let true_kkt =
        Array1::from_shape_vec(5, vec![1.0795966, 1.54808192, 0.3, 1.54213716, 0.5701502]).unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-6);
    assert_eq!(kkt_max, 1.5480819191832393);
}
