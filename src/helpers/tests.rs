use crate::helpers::*;
use ndarray::{Array1, Array2};

#[test]
fn test_singular_matrix() {
    let A = Array2::from_shape_vec((3, 3), vec![4., 0., 0., 0., 0., 0., 0., 0., 3.4]).unwrap();
    let b = Array1::from_shape_vec(3, vec![1., 3., 2.]).unwrap();
    let x = helpers::solve_lin_sys(A.view(), b.view());

    match x {
        Ok(_) => panic!("Should return that the matrix is singular"),
        Err(e) => assert_eq!(e, "Infinitely many solutions or singular matrix"),
    }
}

#[test]
fn test_infinitely_many_solution() {
    let A = Array2::from_shape_vec((2, 2), vec![22., 11., 14., 7.]).unwrap();
    let b = Array1::from_shape_vec(2, vec![44., 28.]).unwrap();
    let x = helpers::solve_lin_sys(A.view(), b.view());

    match x {
        Ok(_) => panic!("Should return that there are infinitely many solutions"),
        Err(e) => assert_eq!(e, "Infinitely many solutions or singular matrix"),
    }
}

#[test]
fn test_linear_system_solver() {
    let A = Array2::from_shape_vec((3, 3), vec![4., 2., 1., 3.4, 2.1, 4.3, 1.0, 0.3, 20.]).unwrap();
    let b = Array1::from_shape_vec(3, vec![1., 3., 2.]).unwrap();
    let x = helpers::solve_lin_sys(A.view(), b.view());

    match x {
        Ok(v) => {
            let truth =
                Array1::from_shape_vec(3, vec![-1.90308498254, 4.24039580908, 0.13154831199])
                    .unwrap();
            test_helpers::assert_array_all_close(v.view(), truth.view(), 1e-9);
        }
        Err(e) => panic!("Unexpected error: {}", e),
    }
}
