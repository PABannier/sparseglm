extern crate ndarray;

use ndarray::{Array1, Array2};

use crate::datafits::*;
use crate::helpers::test_helpers::*;
use crate::penalties::*;
use crate::solver::*;

#[test]
fn test_cd_epoch() {
    let X = Array2::from_shape_vec((2, 3), vec![3.4, -1.2, 2.3, 9.8, -2.7, -0.2]).unwrap();
    let y = Array1::from_shape_vec(2, vec![1.2, -0.9]).unwrap();
    let ws: Vec<usize> = (0..3).collect();

    let mut w = Array1::from_shape_vec(3, vec![1.3, -1.4, 1.5]).unwrap();
    let mut Xw = X.dot(&w);

    let mut datafit = Quadratic::default();
    datafit.initialize(X.view(), y.view());
    let penalty = L1::new(0.3);

    cd_epoch(X.view(), y.view(), &mut w, &mut Xw, &datafit, &penalty, &ws);

    let true_w = Array1::from_shape_vec(3, vec![-0.51752788, -1.24688448, 0.48867352]).unwrap();
    let true_Xw = Array1::from_shape_vec(2, vec![0.86061567, -1.80291985]).unwrap();

    assert_array_all_close(w.view(), true_w.view(), 1e-8);
    assert_array_all_close(Xw.view(), true_Xw.view(), 1e-8);
}

#[test]
fn test_kkt_violation() {
    let X = Array2::from_shape_vec((2, 3), vec![1.6, -1.3, 2.9, 10.8, -3.8, -0.1]).unwrap();
    let y = Array1::from_shape_vec(2, vec![1.4, -0.2]).unwrap();
    let ws: Vec<usize> = (0..3).collect();

    let mut w = Array1::from_shape_vec(3, vec![0.2, -0.3, 1.5]).unwrap();
    let mut Xw = X.dot(&w);

    let mut datafit = Quadratic::default();
    datafit.initialize(X.view(), y.view());
    let penalty = L1::new(0.3);

    #[rustfmt::skip]
    let kkt = kkt_violation(
        X.view(), y.view(), w.view(), Xw.view(), &ws, &datafit, &penalty);
    let kkt = Array1::from_shape_vec(3, kkt).unwrap();
    let true_kkt = Array1::from_shape_vec(3, vec![21.318, 9.044, 5.4395]).unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-8);
}
