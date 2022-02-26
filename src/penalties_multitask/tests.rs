extern crate ndarray;

use ndarray::array;

use crate::penalties_multitask::*;

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

                    assert_eq!(soft_a, $payload.prox.0);
                    assert_eq!(soft_b, $payload.prox.1);
                    assert_eq!(soft_c, $payload.prox.2);
                }

                #[test]
                fn test_subdiff_dist() {
                    let penalty = $payload.penalty;

                    let W = array![[-3.3, 0.1, 3.2], [0.4, 3.2, -3.4], [1.3, 4.3, -0.9]];
                    let grad = array![[0.4, 3.2, -3.4], [0.2, 3.2, -3.], [0.8, -1.2, -2.3]];
                    let ws = array![0, 1, 2];

                    let (subdiff_dist, max_dist) = penalty.subdiff_distance(W.view(), grad.view(), ws.view());

                    assert_eq!(subdiff_dist, $payload.subdiff_dist.0);
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
        prox: (array![0.5870692727288661, -1.6633629393984541, 5.870692727288661], array![0., 0., 0.], array![-77.20780006769976, -0.7441715669175881, -36.278363887232416]),
        subdiff_dist: (array![3.957712409121441, 7.585823111549088, 3.840091062127222], 7.585823111549088),
    },

    l21_2: Payload {
        penalty: L21::new(2.),
        value: 27.629655753614717,
        prox: (array![0.8169182954555414, -2.3146018371240342, 8.169182954555414], array![0., 0., 0.], array![-79.37987504231235, -0.7651072293234926, -37.298977429520264]),
        subdiff_dist: (array![3.9528065593728656, 6.387131216345893, 3.0951877331421045], 6.387131216345893),
    },

    l21_1: Payload {
        penalty: L21::new(1.),
        value: 13.814827876807358,
        prox: (array![1.0084591477277707, -2.857300918562017, 10.084591477277707], array![0.11340887592866315, 0.03780295864288772, 1.209694676572407], array![-81.18993752115618, -0.7825536146617463, -38.14948871476013]),
        subdiff_dist: (array![4.218096709169975, 5.3886661232080515, 2.7340617315080284], 5.3886661232080515),
    },

    blockmcp_075_25: Payload {
        penalty: BlockMCP::new(0.75, 2.5),
        value: 2.109375,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.686149805543993, 4.39089968002003, 2.7147743920996454], 4.686149805543993),
    },

    blockmcp_05_3: Payload {
        penalty: BlockMCP::new(0.5, 3.),
        value: 1.125,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.686149805543993, 4.39089968002003, 2.7147743920996454], 4.686149805543993),
    },

    blockmcp_03_27: Payload {
        penalty: BlockMCP::new(0.3, 2.7),
        value: 0.3645,
        prox: (array![1.2, -3.4, 12.], array![0.3, 0.1, 3.2], array![-83., -0.8, -39.]),
        subdiff_dist: (array![4.686149805543993, 4.39089968002003, 2.7147743920996454], 4.686149805543993),
    },
}
