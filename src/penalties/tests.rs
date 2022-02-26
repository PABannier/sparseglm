extern crate ndarray;

use ndarray::{array, Array1};

use crate::penalties::*;

macro_rules! prox_tests {
    ($($penalty_name:ident: $payload:expr,)*) => {
        $(
            mod $penalty_name {
                use super::*;

                #[test]
                fn test_value() {
                    let a = array![3.4, 2.1, -2.3, -0.3, 4.5];
                    let penalty = $payload.penalty;
                    let val = penalty.value(a.view());
                    assert_eq!(val, $payload.value);
                }

                #[test]
                fn test_prox() {
                    let a = 0.3;
                    let b = 12.4;
                    let c = -49.2;

                    let penalty = $payload.penalty;

                    let soft_a = penalty.prox_op(a, 1. / 0.5);
                    let soft_b = penalty.prox_op(b, 1. / 0.5);
                    let soft_c = penalty.prox_op(c, 1. / 0.5);

                    assert_eq!(soft_a, $payload.prox.0);
                    assert_eq!(soft_b, $payload.prox.1);
                    assert_eq!(soft_c, $payload.prox.2);
                }

                #[test]
                fn test_subdiff_dist() {
                    let w = array![-3.3, 0.1, 3.2];
                    let grad = array![0.4, 3.2, -3.4];
                    let ws = array![0, 1, 2];

                    let penalty = $payload.penalty;
                    let (subdiff_dist, max_dist) = penalty.subdiff_distance(w.view(), grad.view(), ws.view());

                    assert_eq!(subdiff_dist, $payload.subdiff_dist.0);
                    assert_eq!(max_dist, $payload.subdiff_dist.1);
                }
            }
        )*
    }
}

struct Payload<T: Penalty<f64>> {
    penalty: T,
    value: f64,
    prox: (f64, f64, f64),
    subdiff_dist: (Array1<f64>, f64),
}

prox_tests! {
    l1_32: Payload {
        penalty: L1::new(3.2),
        value: 40.32,
        prox: (0., 6., -42.800000000000004),
        subdiff_dist: (array![2.8000000000000003, 6.4, 0.19999999999999973], 6.4)
    },

    l1_2: Payload {
        penalty: L1::new(2.),
        value: 25.2,
        prox: (0., 8.4, -45.2),
        subdiff_dist: (array![1.6, 5.2, 1.4], 5.2)
    },

    l1_1: Payload {
        penalty: L1::new(1.),
        value: 12.6,
        prox: (0., 10.4, -47.2),
        subdiff_dist: (array![0.6, 4.2, 2.4], 4.2)
    },

    l05_32: Payload {
        penalty: L05::new(3.2),
        value: 23.931726579048444,
        prox: (0., 11.45449914422858, -48.741647212023324),
        subdiff_dist: (array![0.4807710121010885, 8.259644256269407, 2.5055728090000837], 8.259644256269407)
    },

    l05_2: Payload {
        penalty: L05::new(2.),
        value: 14.957329111905278,
        prox: (0., 11.818226629068471, -48.91403475649643),
        subdiff_dist: (array![0.1504818825631803, 6.362277660168379, 2.8409830056250525], 6.362277660168379)
    },

    l05_1: Payload {
        penalty: L05::new(1.),
        value: 7.478664555952639,
        prox: (0., 12.112670612996371, -49.05722620426811),
        subdiff_dist: (array![0.12475905871840987, 4.78113883008419, 3.120491502812526], 4.78113883008419)
    },

    mcp_32_3: Payload {
        penalty: MCP::new(3.2, 3.),
        value: 33.38666666666667,
        prox: (0., 12.4, -49.2),
        subdiff_dist: (array![1.7000000000000004, 6.366666666666667, 1.2666666666666664], 6.366666666666667)
    },

    mcp_2_3: Payload {
        penalty: MCP::new(2., 3.),
        value: 18.266666666666666,
        prox: (0., 12.4, -49.2),
        subdiff_dist: (array![0.5000000000000002, 5.166666666666667, 2.466666666666667], 5.166666666666667)
    },

    mcp_1_25: Payload {
        penalty: MCP::new(1., 2.5),
        value: 5.242,
        prox: (0., 12.4, -49.2),
        subdiff_dist: (array![0.4, 4.16, 3.4], 4.16)
    },
}
