extern crate ndarray;

use ndarray::{Array1, Array2};

use crate::helpers::test_helpers::*;
use crate::penalties::*;

#[test]
fn test_value_l1() {
    let a = Array1::from_shape_vec(5, vec![3.4, 2.1, -2.3, -0.3, 4.5]).unwrap();
    let pen = L1 { alpha: 3.2 };
    let val = pen.value(a.view());
    assert_eq!(val, 40.32);
}

#[test]
fn test_prox_l1() {
    let a = 0.3;
    let b = 12.4;
    let c = -49.2;
    let pen = L1 { alpha: 2. };
    let soft_a = pen.prox_op(a, 1. / 0.5);
    let soft_b = pen.prox_op(b, 1. / 0.5);
    let soft_c = pen.prox_op(c, 1. / 0.5);
    assert_eq!(soft_a, 0.0);
    assert_eq!(soft_b, 8.4);
    assert_eq!(soft_c, -45.2);
}

#[test]
fn test_subdiff_dist_l1() {
    let w = Array1::from_shape_vec(3, vec![-3.3, 0.1, 3.2]).unwrap();
    let grad = Array1::from_shape_vec(3, vec![0.4, 3.2, -3.4]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();
    let pen = L1 { alpha: 1. };
    let (subdiff_dist, max_dist) = pen.subdiff_distance(w.view(), grad.view(), ws.view());
    let res = Array1::from_shape_vec(3, vec![0.6, 4.2, 2.4]).unwrap();
    assert_eq!(subdiff_dist, res);
    assert_eq!(max_dist, 4.2);
}

#[test]
fn test_value_l21() {
    let a = Array2::from_shape_vec((3, 3), vec![3.4, 2.1, -2.3, -0.3, 4.5, -0.3, 2.3, 1.2, 3.9])
        .unwrap();
    let pen = L21::new(3.2);
    let val = pen.value(a.view());
    assert_eq!(val, 44.20744920578355);
}

#[test]
fn test_prox_l21() {
    let a = Array1::from_shape_vec(3, vec![1.2, -3.4, 12.]).unwrap();
    let b = Array1::from_shape_vec(3, vec![0.3, 0.1, 3.2]).unwrap();
    let c = Array1::from_shape_vec(3, vec![-83., -0.8, -39.]).unwrap();

    let pen = L21 { alpha: 2. };
    let soft_a = pen.prox_op(a.view(), 1. / 0.5);
    let soft_b = pen.prox_op(b.view(), 1. / 0.5);
    let soft_c = pen.prox_op(c.view(), 1. / 0.5);

    let true_a = Array1::from_shape_vec(3, vec![0.8169183, -2.31460184, 8.16918295]).unwrap();
    let true_b = Array1::from_shape_vec(3, vec![0., 0., 0.]).unwrap();
    let true_c = Array1::from_shape_vec(3, vec![-79.37987504, -0.76510723, -37.29897743]).unwrap();

    assert_array_all_close(soft_a.view(), true_a.view(), 1e-6);
    assert_array_all_close(soft_b.view(), true_b.view(), 1e-6);
    assert_array_all_close(soft_c.view(), true_c.view(), 1e-6);
}

#[test]
fn test_subdiff_dist_l21() {
    let W = Array2::from_shape_vec((3, 3), vec![-3.3, 0.1, 3.2, 0.4, 3.2, -3.4, 1.3, 4.3, -0.9])
        .unwrap();
    let grad = Array2::from_shape_vec((3, 3), vec![0.4, 3.2, -3.4, 0.2, 3.2, -3., 0.8, -1.2, -2.3])
        .unwrap();

    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();
    let pen = L21::new(1.);

    let (subdiff_dist, max_dist) = pen.subdiff_distance(W.view(), grad.view(), ws.view());

    let truth = Array1::from_shape_vec(3, vec![4.21809671, 5.38866612, 2.73406173]).unwrap();
    assert_array_all_close(subdiff_dist.view(), truth.view(), 1e-6);
    assert_eq!(max_dist, 5.3886661232080515);
}
