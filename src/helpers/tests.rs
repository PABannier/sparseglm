use super::cholesky::*;
use ndarray::{array, Array2};

#[test]
fn it_works() {
    let mut mat = array![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]];
    cholesky_factorization(&mut mat);
    let res = array![[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]];
    assert_eq!(mat, res);
}

#[test]
fn returns_error() {
    let mut mat = Array2::<f64>::zeros((3, 3));
    match cholesky_factorization(&mut mat) {
        Some(_) => println!("Good"),
        None => panic!("Should return an error but didn't"),
    }
}
