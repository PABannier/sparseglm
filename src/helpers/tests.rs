use super::cholesky::*;
use super::test_helpers::assert_array_all_close;
use ndarray::{array, Array2};

#[test]
fn cholesky_works() {
    let mat = array![[4., 12., -16.], [12., 37., -43.], [-16., -43., 98.]];
    let res = cholesky_factorization(&mat).unwrap();
    let ans = array![[2., 0., 0.], [6., 1., 0.], [-8., 5., 3.]];
    assert_eq!(res, ans);
}

#[test]
fn cholesky_returns_error() {
    let mat = Array2::<f64>::zeros((3, 3));
    match cholesky_factorization(&mat) {
        Err(_) => println!("Good"),
        Ok(_) => panic!("Should return an error but didn't"),
    }
}

#[test]
fn forward_substitution_one_works() {
    let mat = array![[3., 0., 0.], [1., 2., 0.], [-3., 42., 12.]];
    let res = forward_substitution_one(&mat, 3).unwrap();
    let ans = array![0.333333, 0.333333, -1.];
    assert_array_all_close(res.view(), ans.view(), 1e-5);
}

#[test]
fn forward_substitution_one_returns_error() {
    let mat = array![[0., 0., 0.], [1., 2., 0.], [0., 3.4, 2.3]];
    match forward_substitution_one(&mat, 3) {
        Err(_) => println!("Good"),
        Ok(_) => panic!("Should return an error but didn't"),
    }
}

#[test]
fn backward_substitution_works() {
    let mat = array![[3., 2., 1.], [0., 1., 1.], [0., 0., 2.]];
    let mut b = array![1., 2., 3.];
    let res = backward_substitution(mat.view(), &mut b).unwrap();
    let ans = array![-0.5, 0.5, 1.5];
    assert_array_all_close(res.view(), ans.view(), 1e-5);
}

#[test]
fn backward_substitution_returns_error() {
    let mat = array![[3., 0., 1.], [0., 0., 1.], [0., 0., 2.3]];
    let mut b = array![1., 1., 1.];
    match backward_substitution(mat.view(), &mut b) {
        Err(_) => println!("Good"),
        Ok(_) => panic!("Should return an error but didn't"),
    }
}

#[test]
fn solve_lin_sys_one_by_cholesky_works() {
    let mat = array![[26., 8., 15.], [8., 14., 5.], [15., 5., 14.]];
    let res = solve_lin_sys_one_by_cholesky(&mat).unwrap();
    let ans = array![-0.0225, 0.0575, 0.075];
    assert_array_all_close(res.view(), ans.view(), 1e-5);
}

#[test]
fn solve_lin_sys_one_by_cholesky_returns_error() {
    let mat = array![[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]];
    match solve_lin_sys_one_by_cholesky(&mat) {
        Ok(_) => panic!("Should return an error"),
        Err(_) => println!("Good!"),
    }
}
