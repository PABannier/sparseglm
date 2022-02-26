extern crate ndarray;

use ndarray::array;

use crate::{helpers::test_helpers::assert_array_all_close, penalties_multitask::*};

macro_rules! prox_tests {
    ($($penalty_name:ident: $payload:expr,)*) => {
        $(
            mod $penalty_name {
                use super::*;

                #[test]
                fn test_value() {
                    let penalty = $payload.penalty;

                    let a = array![[3.4, 2.1, -2.3], [-0.3, 4.5, -0.3], [2.3, 1.2, 3.9]];
                    let val = penalty.value(a.view());

                    assert_eq!(val, $payload.value);
                }

                #[test]
                fn test_prox() {
                    let penalty = $payload.penalty;

                    let a = array![1.2, -3.4, 12.];
                    let b = array![0.3, 0.1, 3.2];
                    let c = array![-83., -0.8, -39.];

                    let soft_a = penalty.prox_op(a.view(), 1. / 0.5);
                    let soft_b = penalty.prox_op(b.view(), 1. / 0.5);
                    let soft_c = penalty.prox_op(c.view(), 1. / 0.5);

                    assert_array_all_close(soft_a.view(), $payload.prox.0.view(), 1e-6);
                    assert_array_all_close(soft_b.view(), $payload.prox.1.view(), 1e-6);
                    assert_array_all_close(soft_c.view(), $payload.prox.2.view(), 1e-6);
                }

                #[test]
                fn test_subdiff_dist() {
                    let penalty = $payload.penalty;

                    let W = array![[-3.3, 0.1, 3.2], [0.4, 3.2, -3.4], [1.3, 4.3, -0.9]];
                    let grad = array![[0.4, 3.2, -3.4], [0.2, 3.2, -3.], [0.8, -1.2, -2.3]];
                    let ws = array![0, 1, 2];

                    let (subdiff_dist, max_dist) = penalty.subdiff_distance(W.view(), grad.view(), ws.view());

                    assert_array_all_close(subdiff_dist.view(), $payload.subdiff_dist.0.view(), 1e-6);
                    assert_eq!(max_dist, $payload.subdiff_dist.1);
                }
            }
        )*
    }
}

struct Payload<T: PenaltyMultiTask<f64>> {
    penalty: T,
    value: f64,
    prox: (Array1<f64>, Array1<f64>, Array1<f64>),
    subdiff_dist: (Array1<f64>, f64),
}

prox_tests! {
    l21_32: Payload {
        penalty: L21::new(3.2),
        value: 44.20744920578355,
        prox: (array![0.58706927, -1.66336294, 5.87069273], array![0., 0., 0.], array![-77.20780007, -0.74417157, -36.27836389]),
        subdiff_dist: (array![3.95771241, 7.58582311, 3.84009106], 7.58582311),
    },

    l21_2: Payload {
        penalty: L21::new(2.),
        value: 27.629655753614717,
        prox: (array![0.8169183, -2.31460184, 8.16918295], array![0., 0., 0.], array![-79.37987504, -0.76510723, -37.29897743]),
        subdiff_dist: (array![3.95280656, 6.38713122, 3.09518773], 6.38713122),
    },

    l21_1: Payload {
        penalty: L21::new(1.),
        value: 13.814827876807358,
        prox: (array![1.00845915, -2.85730092, 10.08459148], array![0.11340888, 0.03780296, 1.20969468], array![-81.18993752, -0.78255361, -38.14948871]),
        subdiff_dist: (array![4.21809671, 5.38866612, 2.73406173], 5.3886661232080515),
    },

    blockmcp_075_25: Payload {
        penalty: BlockMCP::new(0.75, 2.5),
        value: 2.109375,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.68614981, 4.39089968, 2.71477439], 4.68614981),
    },

    blockmcp_05_3: Payload {
        penalty: BlockMCP::new(0.5, 3.),
        value: 1.125,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.21809671, 5.38866612, 2.73406173], 5.3886661232080515),
    },

    blockmcp_03_27: Payload {
        penalty: BlockMCP::new(0.3, 2.7),
        value: 0.3645,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.21809671, 5.38866612, 2.73406173], 5.3886661232080515),
    },
}
