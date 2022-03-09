use crate::datafits::*;
use crate::datafits_multitask::*;
use crate::datasets::{DatasetBase, DesignMatrix};
use crate::helpers::test_helpers::{assert_array2d_all_close, assert_array_all_close};
use crate::penalties::*;
use crate::penalties_multitask::*;

use super::{BCDSolver, CDSolver, Solver};
use ndarray::{array, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Ix1, Ix2, OwnedRepr};

macro_rules! estimator_test {
    ($($penalty_name:ident: $payload:expr,)*) => {
        $(
            #[test]
            fn $penalty_name() {
                let x = $payload.design_matrix;
                let y = $payload.target;
                let dataset = DatasetBase::from((x, y));

                let mut datafit = $payload.datafit;
                let penalty = $payload.penalty;
                let solver = Solver::default();

                let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
                assert_array_all_close(coefficients.view(), $payload.truth, 1e-5);

            }
        )*
    }
}

macro_rules! multi_task_estimator_test {
    ($($penalty_name:ident: $payload:expr,)*) => {
        $(
            #[test]
            fn $penalty_name() {
                let x = $payload.design_matrix;
                let y = $payload.target;
                let dataset = DatasetBase::from((x, y));

                let mut datafit = $payload.datafit;
                let penalty = $payload.penalty;
                let solver = Solver::default();

                let coefficients = solver.solve_multi_task(&dataset, &mut datafit, &penalty).unwrap();
                assert_array2d_all_close(coefficients.view(), $payload.truth, 1e-5);

            }
        )*
    }
}

struct Payload<
    'a,
    DM: DesignMatrix<Elem = f64>,
    P: Penalty<f64>,
    DF: Datafit<f64, DM, ArrayBase<OwnedRepr<f64>, Ix1>>,
> {
    design_matrix: DM,
    target: Array1<f64>,
    penalty: P,
    datafit: DF,
    truth: ArrayView1<'a, f64>,
}

estimator_test! {
    // lasso_dense: Payload {
    //     design_matrix: array![[3., 2., 1.], [0.5, 3.4, 1.2]],
    //     target: array![1.2, 4.5],
    //     penalty: L1::new(1.),
    //     datafit: Quadratic::new(),
    //     truth: array![0., 1.00899743, 0.].view(),
    // },

    // elastic_net_dense: Payload {
    //     design_matrix: array![[3.1, -3.4, 0.3], [0.9, -0.01, 2.3]],
    //     target: array![-0.3, 0.08],
    //     penalty: L1PlusL2::new(1., 0.4),
    //     datafit: Quadratic::new(),
    //     truth: array![0.,  0.01717855,  0.].view(),
    // },

    // mcp_dense: Payload {
    //     design_matrix: array![[3., 2., 1.], [0.5, 3.4, 1.2]],
    //     target: array![1.2, 4.5],
    //     penalty: MCP::new(0.3, 2.),
    //     datafit: Quadratic::new(),
    //     truth: array![-0.52009245, 1.39490436, 0.].view(),
    // },
}

struct MultiTaskPayload<
    'a,
    DM: DesignMatrix<Elem = f64>,
    P: MultiTaskPenalty<f64>,
    DF: MultiTaskDatafit<f64, DM, ArrayBase<OwnedRepr<f64>, Ix2>>,
> {
    design_matrix: DM,
    target: Array2<f64>,
    penalty: P,
    datafit: DF,
    truth: ArrayView2<'a, f64>,
}

multi_task_estimator_test! {
    multi_task_lasso_dense: MultiTaskPayload {
        design_matrix: array![[0.3, 4.4, 1.2], [-1.2, 0.3, 1.2]],
        target: array![[0.3, 1.2, 0.2, -0.3], [1.3, 4.2, 1., -0.3]],
        penalty: L21::new(1.7),
        datafit: QuadraticMultiTask::new(),
        truth: array![[-0.08312622, -0.2596302, -0.06513197, 0.0102676 ], [0., 0.,  0.,  0.], [0.31036133, 1.05129801, 0.23226053, -0.12021403]].view(),
    },

    multi_task_elastic_net_dense: MultiTaskPayload {
        design_matrix: array![[0.3, 4.4, 1.2], [-1.2, 0.3, 1.2]],
        target: array![[0.3, 1.2, 0.2, -0.3], [1.3, 4.2, 1., -0.3]],
        penalty: BlockL1PlusL2::new(1.7, 0.3),
        datafit: QuadraticMultiTask::new(),
        truth: array![[-0.23643401, -0.75067860, -0.18362717,  0.04140032], [ 0.01676151,  0.080385968,  0.00939374, -0.030116  ], [ 0.24411237,  0.8109393,  0.18481232, -0.07858179]].view(),
    },

    // multi_task_mcp_dense: MultiTaskPayload {
    //     design_matrix: array![[0.3, 4.4, 1.2], [-1.2, 0.3, 1.2]],
    //     target: array![[0.3, 1.2, 0.2, -0.3], [1.3, 4.2, 1., -0.3]],
    //     penalty: BlockMCP::new(0.89, 2.5),
    //     datafit: QuadraticMultiTask::new(),
    //     truth: array![[]],
    // },
}
