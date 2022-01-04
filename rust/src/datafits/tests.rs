extern crate ndarray;
extern crate num;

use ndarray::{Array1, Array2};

use crate::datafits::*;
use crate::helpers::test_helpers::*;

#[test]
fn test_initialization_quadratic() {
    let X = Array2::from_shape_vec((2, 3), vec![3.4, 2.1, 2.3, 3.4, -1.2, 0.2]).unwrap();
    let y = Array1::from_shape_vec(2, vec![-3.4, 2.1]).unwrap();
    let mut df = Quadratic::default();
    df.initialize(X.view(), y.view());
    let Xty = Array1::from_shape_vec(3, vec![-4.42, -9.66, -7.4]).unwrap();
    let lipschitz = Array1::from_shape_vec(3, vec![11.56, 2.925, 2.665]).unwrap();
    assert_array_all_close(Xty.view(), df.get_Xty().view(), 1e-8);
    assert_array_all_close(lipschitz.view(), df.get_lipschitz().view(), 1e-8);
}

#[test]
fn test_value_quadratic() {
    let X = Array2::from_shape_vec((2, 3), vec![3.6, 1.1, 2.2, 3.4, -1.2, 0.2]).unwrap();
    let y = Array1::from_shape_vec(2, vec![-3.3, 2.7]).unwrap();
    let w = Array1::from_shape_vec(3, vec![-3.2, -0.21, 2.3]).unwrap();
    let Xw = X.dot(&w);
    let df = Quadratic::default();
    let val = df.value(y.view(), w.view(), Xw.view());
    assert_eq!(val, 44.27107624999999);
}

#[test]
fn test_gradient_quadratic() {
    let X = Array2::from_shape_vec((2, 3), vec![3.0, 1.1, 3.2, 3.4, -1.2, 0.2]).unwrap();
    let y = Array1::from_shape_vec(2, vec![-3.3, 2.4]).unwrap();
    let w = Array1::from_shape_vec(3, vec![-3.2, -0.25, 3.3]).unwrap();
    let Xw = X.dot(&w);
    let mut df = Quadratic::default();
    df.initialize(X.view(), y.view());
    let grad = df.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), 1);
    assert_eq!(grad, 9.583749999999998);
}