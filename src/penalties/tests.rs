extern crate ndarray;

use ndarray::Array1;

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
    let soft_a = pen.prox_op(a, 1. / 0.5, 0);
    let soft_b = pen.prox_op(b, 1. / 0.5, 0);
    let soft_c = pen.prox_op(c, 1. / 0.5, 0);
    assert_eq!(soft_a, 0.0);
    assert_eq!(soft_b, 8.4);
    assert_eq!(soft_c, -45.2);
}

#[test]
fn test_subdiff_dist_l1() {
    let w = Array1::from_shape_vec(3, vec![-3.3, 0.1, 3.2]).unwrap();
    let grad = Array1::from_shape_vec(3, vec![0.4, 3.2, -3.4]).unwrap();
    let ws: Vec<usize> = (0..3).collect();
    let pen = L1 { alpha: 1. };
    let (subdiff_dist, max_dist) = pen.subdiff_distance(w.view(), grad.view(), &ws);
    let subdiff_dist = Array1::from_shape_vec(3, subdiff_dist).unwrap();
    let res = Array1::from_shape_vec(3, vec![0.6, 4.2, 2.4]).unwrap();
    assert_eq!(subdiff_dist, res);
    assert_eq!(max_dist, 4.2);
}
