use ndarray::{array, Array1};

use crate::datafits::single_task::*;
use crate::datasets::*;
use crate::helpers::test_helpers::*;

macro_rules! datafit_tests {
    ($($datafit_name:ident: $payload:expr,)*) => {
        $(
            mod $datafit_name {
                use super::*;

                #[test]
                fn test_initialization() {
                    let x = array![[3.4, 2.1, 2.3], [3.4, -1.2, 0.2]];
                    let y = array![-3.4, 2.1];

                    let dataset = DenseDataset::from((x, y));
                    let mut df = $payload.datafit;
                    df.initialize(&dataset);

                    assert_array_all_close($payload.lipschitz.view(), df.lipschitz.view(), 1e-8);
                }

                #[test]
                fn test_initialization_sparse() {
                    let indptr = array![0, 2, 3, 6];
                    let indices = array![0, 2, 2, 0, 1, 2];
                    let data = array![1., 2., 3., 4., 5., 6.];
                    let x_sparse = CSCArray::new(data.view(), indices.view(), indptr.view());

                    let x = array![[1., 0., 4.], [0., 0., 5.], [2., 3., 6.]];
                    let y = array![1., 3., 2.];
                    let dataset = DenseDataset::from((x, y));
                    let y = dataset.targets().try_single_target().unwrap();
                    let dataset_sparse = SparseDataset::from((x_sparse, y));

                    let mut df = $payload.datafit;
                    let mut df_sparse = $payload.datafit;
                    df.initialize(&dataset);
                    df_sparse.initialize(&dataset_sparse);

                    assert_array_all_close(df.lipschitz.view(), df_sparse.lipschitz.view(), 1e-8);
                }

                #[test]
                fn test_value() {
                    let x = array![[3.6, 1.1, 2.2], [3.4, -1.2, 0.2]];
                    let y = array![-3.3, 2.7];
                    let w = array![-3.2, -0.21, 2.3];
                    let xw = x.dot(&w);
                    let dataset = DenseDataset::from((x, y));

                    let df = $payload.datafit;
                    let val = df.value(&dataset, xw.view());
                    assert_eq!(val, $payload.val);
                }

                #[test]
                fn test_gradient() {
                    let x = array![[3.0, 1.1, 3.2], [3.4, -1.2, 0.2]];
                    let y = array![-3.3, 2.4];
                    let w = array![-3.2, -0.25, 3.3];
                    let xw = x.dot(&w);

                    let dataset = DenseDataset::from((x, y));

                    let mut df = $payload.datafit;
                    df.initialize(&dataset);
                    let grad = df.gradient_j(&dataset, xw.view(), 1);
                    assert_eq!(grad, $payload.grad);
                }

                #[test]
                fn test_gradient_sparse() {
                    let indptr = array![0, 2, 3, 6];
                    let indices = array![0, 2, 2, 0, 1, 2];
                    let data = array![1., 2., 3., 4., 5., 6.];
                    let x_sparse = CSCArray::new(data.view(), indices.view(), indptr.view());

                    let x = array![[1., 0., 4.], [0., 0., 5.], [2., 3., 6.]];
                    let y = array![1., 3., 2.];
                    let w = array![-3.2, -0.25, 3.3];

                    let xw = x.dot(&w);

                    let dataset = DenseDataset::from((x, y));
                    let y = dataset.targets().try_single_target().unwrap();
                    let dataset_sparse = SparseDataset::from((x_sparse, y));

                    let mut df = $payload.datafit;
                    let mut df_sparse = $payload.datafit;

                    df.initialize(&dataset);
                    df_sparse.initialize(&dataset_sparse);

                    let grad = df.gradient_j(&dataset, xw.view(), 1);
                    let grad_sparse = df_sparse.gradient_j(&dataset, xw.view(), 1);
                    assert_eq!(grad, grad_sparse);
                }

            }
        )*
    };
}

struct Payload<DF> {
    datafit: DF,
    val: f64,
    grad: f64,
    lipschitz: Array1<f64>,
}

datafit_tests! {
    quadratic: Payload {
        datafit: Quadratic::<f64>::new(),
        val: 44.27107624999999,
        grad: 9.583749999999998,
        lipschitz: array![11.56, 2.925, 2.665],
    },

    logistic: Payload {
        datafit: Logistic::<f64>::new(),
        val: 13.726800000129309,
        grad: 3.0835776120075256,
        lipschitz: array![2.89, 0.73125, 0.66625],
    },
}
