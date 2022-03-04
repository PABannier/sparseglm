use super::cholesky::*;
use ndarray::{array, Array2};

#[test]
fn it_works() {
    let mut mat = array![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]];
    let res = cholesky_factorization(&mat).unwrap();
    let ans = array![[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]];
    assert_eq!(res, ans);
}

#[test]
fn returns_error() {
    let mat = Array2::<f64>::zeros((3, 3));
    match cholesky_factorization(&mat) {
        Err(_) => println!("Good"),
        Ok(_) => panic!("Should return an error but didn't"),
    }
}
