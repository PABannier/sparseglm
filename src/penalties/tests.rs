extern crate ndarray;

use ndarray::{array, Array1};

use crate::penalties::*;

#[test]
fn test_value_l1() {
    let a = array![3.4, 2.1, -2.3, -0.3, 4.5];
    let pen = L1::new(3.2);
    let val = pen.value(a.view());
    assert_eq!(val, 40.32);
}

#[test]
fn test_prox_l1() {
    let a = 0.3;
    let b = 12.4;
    let c = -49.2;
    let pen = L1::new(2.);
    let soft_a = pen.prox_op(a, 1. / 0.5);
    let soft_b = pen.prox_op(b, 1. / 0.5);
    let soft_c = pen.prox_op(c, 1. / 0.5);
    assert_eq!(soft_a, 0.0);
    assert_eq!(soft_b, 8.4);
    assert_eq!(soft_c, -45.2);
}

#[test]
fn test_subdiff_dist_l1() {
    let w = array![-3.3, 0.1, 3.2];
    let grad = array![0.4, 3.2, -3.4];
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();
    let pen = L1::new(1.);
    let (subdiff_dist, max_dist) = pen.subdiff_distance(w.view(), grad.view(), ws.view());
    let res = array![0.6, 4.2, 2.4];
    assert_eq!(subdiff_dist, res);
    assert_eq!(max_dist, 4.2);
}

#[test]
fn test_value_l05() {
    let a = array![3.4, 2.1, -2.3, -0.3, 4.5];
    let pen = L05::new(3.2);
    let val = pen.value(a.view());
    assert_eq!(val, 23.931726579048444);
}

#[test]
fn test_prox_l05() {
    let a = 0.3;
    let b = 12.4;
    let c = -49.2;
    let pen = L05::new(2.);
    let soft_a = pen.prox_op(a, 1. / 0.5);
    let soft_b = pen.prox_op(b, 1. / 0.5);
    let soft_c = pen.prox_op(c, 1. / 0.5);
    assert_eq!(soft_a, 0.);
    assert_eq!(soft_b, 11.818226629068471);
    assert_eq!(soft_c, -48.91403475649643);
}

#[test]
fn test_subdiff_dist_l05() {
    let w = array![-3.3, 0.1, 3.2];
    let grad = array![0.4, 3.2, -3.4];
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();
    let pen = L05::new(1.);
    let (subdiff_dist, max_dist) = pen.subdiff_distance(w.view(), grad.view(), ws.view());
    let res = array![0.12475905871840987, 4.78113883008419, 3.120491502812526];
    assert_eq!(subdiff_dist, res);
    assert_eq!(max_dist, 4.78113883008419);
}

#[test]
fn test_value_mcp() {
    let a = array![3.4, 2.1, -2.3, -0.3, 4.5];
    let pen = MCP::new(3.2, 3.);
    let val = pen.value(a.view());
    assert_eq!(val, 33.38666666666667);
}

#[test]
fn test_prox_mcp() {
    let a = 0.3;
    let b = 12.4;
    let c = -49.2;
    let pen = MCP::new(2., 3.);
    let pen_a = pen.prox_op(a, 1. / 0.5);
    let pen_b = pen.prox_op(b, 1. / 0.5);
    let pen_c = pen.prox_op(c, 1. / 0.5);
    assert_eq!(pen_a, 0.);
    assert_eq!(pen_b, 12.4);
    assert_eq!(pen_c, -49.2);
}

#[test]
fn test_subdiff_dist_mcp() {
    let w = array![-3.2, 0.5, -1.2];
    let grad = array![0.4, 2.3, -3.2];
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();
    let pen = MCP::new(1., 2.5);
    let (subdiff_dist, max_dist) = pen.subdiff_distance(w.view(), grad.view(), ws.view());
    let res = array![0.4, 3.0999999999999996, 3.72];
    assert_eq!(subdiff_dist, res);
    assert_eq!(max_dist, 3.72);
}
