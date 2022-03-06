use crate::datafits::*;
use crate::datasets::DatasetBase;
use crate::datasets::DesignMatrix;
use crate::helpers::test_helpers::assert_array_all_close;
use crate::penalties::*;

use super::{CDSolver, Solver};
use ndarray::{array, Array1, ArrayBase, ArrayView1, Ix1, OwnedRepr};

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
    lasso_dense: Payload {
        design_matrix: array![[3., 2., 1.], [0.5, 3.4, 1.2]],
        target: array![1.2, 4.5],
        penalty: L1::new(1.),
        datafit: Quadratic::new(),
        truth: array![0., 1.00899743, 0.].view(),
    },

    elastic_net_dense: Payload {
        design_matrix: array![[3.1, -3.4, 0.3], [0.9, -0.01, 2.3]],
        target: array![-0.3, 0.08],
        penalty: L1PlusL2::new(1., 0.4),
        datafit: Quadratic::new(),
        truth: array![0.,  0.01717855,  0.].view(),
    },

    mcp_dense: Payload {
        design_matrix: array![[3., 2., 1.], [0.5, 3.4, 1.2]],
        target: array![1.2, 4.5],
        penalty: MCP::new(0.3, 2.),
        datafit: Quadratic::new(),
        truth: array![-0.52009245, 1.39490436, 0.].view(),
    },
}
