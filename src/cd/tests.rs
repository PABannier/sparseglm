use ndarray::{Array1, Array2};

use crate::cd::*;
use crate::datafits::single_task::*;
use crate::datasets::csc_array::CSCArray;
use crate::helpers::test_helpers::*;
use crate::penalties::*;

#[test]
fn test_kkt_violation() {
    let X = Array2::from_shape_vec((2, 3), vec![1.6, -1.3, 2.9, 10.8, -3.8, -0.1]).unwrap();
    let y = Array1::from_shape_vec(2, vec![1.4, -0.2]).unwrap();
    let ws = Array1::from_shape_vec(3, (0..3).collect()).unwrap();

    let w = Array1::from_shape_vec(3, vec![0.2, -0.3, 1.5]).unwrap();
    let Xw = X.dot(&w);

    let dataset = DatasetBase::from((X, y));

    let mut datafit = Quadratic::new();
    datafit.initialize(&dataset);

    let penalty = L1::new(0.3);

    let (kkt, kkt_max) =
        kkt_violation(&dataset, w.view(), Xw.view(), ws.view(), &datafit, &penalty);
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

    let dataset = DatasetBase::from((X, y));

    let mut datafit = Quadratic::new();
    datafit.initialize(&dataset);
    let penalty = L1::new(0.3);

    let (kkt, kkt_max) =
        kkt_violation(&dataset, w.view(), Xw.view(), ws.view(), &datafit, &penalty);
    let true_kkt =
        Array1::from_shape_vec(5, vec![0.95174179, 0.34320058, 0.3, 1.07978458, 0.12640847])
            .unwrap();

    assert_array_all_close(kkt.view(), true_kkt.view(), 1e-8);
    assert_eq!(kkt_max, 1.0797845792515859);
}
