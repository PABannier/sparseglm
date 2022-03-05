use crate::datasets::DatasetBase;
use crate::{datafits::*, datafits_multitask::QuadraticMultiTask};
use crate::{penalties::*, penalties_multitask::L21};

use super::{BCDSolver, CDSolver, Solver};
use ndarray::array;

#[test]
fn test_lasso() {
    let x = array![[3., 2., 1.], [0.5, 3.4, 1.2]];
    let y = array![1.2, 4.5];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = Quadratic::new();
    let penalty = L1::new(0.1);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
    assert_eq!(
        coefficients,
        array![-0.4798204158784847, 1.3621219281657122, 0.]
    );
}

#[test]
fn test_multi_task_lasso() {
    let x = array![[-0.3, -2.1, 1.4, 3.4], [3.4, 1.2, -0.3, -0.1]];
    let y = array![[0.3, 0.2], [-0.2, -0.1]];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = QuadraticMultiTask::new();
    let penalty = L21::new(0.07);
    let solver = Solver::default();

    let coefficients = solver
        .solve_multi_task(&dataset, &mut datafit, &penalty)
        .unwrap();
    assert_eq!(
        coefficients,
        array![
            [-0.04664696969076872, -0.023139825340769624],
            [0., 0.],
            [0., 0.],
            [0.07437213388897676, 0.050151821872412336]
        ]
    );
}

// Add one test for sparse matrices

// Add one test for multitask lasso (dense and sparse)

// Add one test for estimators

// Write it with macros
