extern crate ndarray;

use ndarray::{Array1, Array2};

use crate::datafits::*;
use crate::dataset::*;
use crate::helpers::test_helpers::*;
use crate::penalties::*;
use crate::solver::*;

#[test]
fn test_cd_epoch() {
    let X = Array2::from_shape_vec((2, 3), vec![3.4, -1.2, 2.3, 9.8, -2.7, -0.2]).unwrap();
    let y = Array1::from_shape_vec(2, vec![1.2, -0.9]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let mut w = Array1::from_shape_vec(3, vec![1.3, -1.4, 1.5]).unwrap();
    let mut Xw = X.dot(&w);

    let mut datafit = Quadratic::default();
    datafit.initialize(X.view(), y.view());
    let penalty = L1::new(0.3);

    cd_epoch(X.view(), &mut w, &mut Xw, &datafit, &penalty, ws.view());

    let true_w = Array1::from_shape_vec(3, vec![-0.51752788, -1.24688448, 0.48867352]).unwrap();
    let true_Xw = Array1::from_shape_vec(2, vec![0.86061567, -1.80291985]).unwrap();

    assert_array_all_close(w.view(), true_w.view(), 1e-8);
    assert_array_all_close(Xw.view(), true_Xw.view(), 1e-8);
}

#[test]
fn test_cd_epoch_sparse() {
    let indptr = Array1::from_shape_vec(4, vec![0, 2, 3, 6]).unwrap();
    let indices = Array1::from_shape_vec(6, vec![0, 2, 2, 0, 1, 2]).unwrap();
    let data = Array1::from_shape_vec(6, vec![1., 2., 3., 4., 5., 6.]).unwrap();
    let X = CSCArray::new(data.view(), indices.view(), indptr.view());
    let X_full = Array2::from_shape_vec((3, 3), vec![1., 0., 2., 0., 0., 3., 4., 5., 6.]).unwrap();
    let y = Array1::from_shape_vec(3, vec![1.2, -0.9, 0.1]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let mut w = Array1::from_shape_vec(3, vec![2.1, -0.9, 3.4]).unwrap();
    let mut Xw = X_full.dot(&w);

    let mut datafit = Quadratic::default();
    datafit.initialize_sparse(&X, y.view());
    let penalty = L1::new(0.7);

    cd_epoch_sparse(&X, &mut w, &mut Xw, &datafit, &penalty, ws.view());

    let true_w = Array1::from_shape_vec(3, vec![-8.7, -1.53333333, 2.75844156]).unwrap();
    let true_Xw = Array1::from_shape_vec(3, vec![-4.46623377, 6.99220779, -3.04935065]).unwrap();

    assert_array_all_close(w.view(), true_w.view(), 1e-8);
    assert_array_all_close(Xw.view(), true_Xw.view(), 1e-8);
}

#[test]
fn test_kkt_violation() {
    let X = Array2::from_shape_vec((2, 3), vec![1.6, -1.3, 2.9, 10.8, -3.8, -0.1]).unwrap();
    let y = Array1::from_shape_vec(2, vec![1.4, -0.2]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let w = Array1::from_shape_vec(3, vec![0.2, -0.3, 1.5]).unwrap();
    let Xw = X.dot(&w);

    let mut datafit = Quadratic::default();
    datafit.initialize(X.view(), y.view());
    let penalty = L1::new(0.3);

    let (kkt, kkt_max) =
        kkt_violation(X.view(), w.view(), Xw.view(), ws.view(), &datafit, &penalty);
    let true_kkt = Array1::from_shape_vec(3, vec![21.318, 9.044, 5.4395]).unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-8);
    assert_eq!(kkt_max, 21.318);
}

#[test]
fn test_kkt_violation_sparse() {
    let data =
        Array1::from_shape_vec(4, vec![0.74272001, 0.95888754, 0.8886366, 0.19782359]).unwrap();
    let indices = Array1::from_shape_vec(4, vec![0, 1, 0, 0]).unwrap();
    let indptr = Array1::from_shape_vec(6, vec![0, 1, 2, 2, 3, 4]).unwrap();

    let X = CSCArray::new(data.view(), indices.view(), indptr.view());
    let y = Array1::from_shape_vec(3, vec![0.3, -2.3, 0.8]).unwrap();
    let ws = Array1::from_shape_vec(5, (0..5).collect()).unwrap();

    let w = Array1::from_shape_vec(5, vec![0.2, -0.3, 1.3, 3.4, -1.2]).unwrap();
    let Xw = Array1::from_shape_vec(3, vec![2.93252013, -0.28766626, 0.]).unwrap();

    let mut datafit = Quadratic::default();
    datafit.initialize_sparse(&X, y.view());
    let penalty = L1::new(0.3);

    let (kkt, kkt_max) =
        kkt_violation_sparse(&X, w.view(), Xw.view(), ws.view(), &datafit, &penalty);
    let true_kkt =
        Array1::from_shape_vec(5, vec![0.95174179, 0.34320058, 0.3, 1.07978458, 0.12640847])
            .unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-8);
    assert_eq!(kkt_max, 1.0797845792515859);
}
