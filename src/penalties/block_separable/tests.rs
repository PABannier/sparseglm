use ndarray::array;

use crate::penalties::block_separable::*;

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

                    let soft_a = penalty.prox(a.view(), 1. / 0.5);
                    let soft_b = penalty.prox(b.view(), 1. / 0.5);
                    let soft_c = penalty.prox(c.view(), 1. / 0.5);

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

struct Payload<T: MultiTaskPenalty<f64>> {
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

    block_l1_plus_l2_1_05: Payload {
        penalty: BlockL1PlusL2::new(1., 0.5),
        value: 22.814913938403677,
        prox: (array![0.5521147869319426, -1.5643252296405041, 5.521147869319426], array![0.10335221898216579, 0.03445073966072193, 1.1024236691431017], array![-41.04748438028904, -0.39563840366543657, -19.28737217869003]),
        subdiff_dist: (array![3.9153649523984124, 7.229241744227194, 3.5582002492326654], 7.229241744227194)
    },

    block_l1_plus_l2_27_03: Payload {
        penalty: BlockL1PlusL2::new(2.7, 0.3),
        value: 71.32036058021396,
        prox: (array![0.21858826561914108, -0.619333419254233, 2.185882656191411], array![0.031142508264062167, 0.010380836088020724, 0.33218675481666315], array![-17.057290667810985, -0.16440762089456373, -8.014871518609981]),
        subdiff_dist: (array![7.944539735963811, 14.049442151573732, 9.410536326066207], 14.049442151573732)
    },

    block_l1_plus_l2_09_09: Payload {
        penalty: BlockL1PlusL2::new(0.9, 0.9),
        value: 14.05336058021396,
        prox: (array![0.8854677200504188, -2.50882520680952, 8.85467720050419], array![0.12615355042560775, 0.04205118347520259, 1.345637871206483], array![-69.0964825357089, -0.6659901931152665, -32.46702191436925]),
        subdiff_dist: (array![4.1392622827449825, 5.620015296905382, 2.7875123700916884], 5.620015296905382)
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
