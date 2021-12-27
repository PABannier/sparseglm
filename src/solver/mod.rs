extern crate ndarray;
extern crate num;

use crate::datafits::Datafit;
use crate::penalties::Penalty;
use ndarray::{s, Array1, ArrayView1, ArrayView2};
use num::Float;

pub fn soft_thresholding<T: Float>(x: T, threshold: T) -> T {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        T::zero()
    }
}

pub fn get_max_arr<T: Float>(arr: ArrayView1<T>) -> T {
    let mut max_val = T::neg_infinity();
    for j in 0..arr.len() {
        if arr[j] > max_val {
            max_val = arr[j];
        }
    }
    max_val
}

pub fn construct_grad<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    w: ArrayView1<T>,
    Xw: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: &D,
) -> Array1<T> {
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
    }
    grad
}

pub fn cd_epoch<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    w: &mut Array1<T>,
    Xw: &mut Array1<T>,
    datafit: &D,
    penalty: &P,
    ws: ArrayView1<usize>,
) {
    let n_samples = X.shape()[0];
    let lipschitz = datafit.get_lipschitz();

    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        let old_w_j = w[j];
        let grad_j = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
        w[j] = penalty.prox_op(old_w_j - grad_j / lipschitz[j], T::one() / lipschitz[j], j);
        if w[j] != old_w_j {
            for i in 0..n_samples {
                Xw[i] = Xw[i] + (w[j] - old_w_j) * Xj[i];
            }
        }
    }
}

pub fn solver<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    datafit: &D,
    penalty: &P,
    w: &mut Array1<T>,
    Xw: &mut Array1<T>,
    max_iter: usize,
    max_epochs: usize,
    p0: usize,
    tol: T,
    verbose: bool,
) {
    let n_features = X.shape()[1];
    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    for epoch in 0..max_epochs {
        cd_epoch(
            X.view(),
            y.view(),
            w,
            Xw,
            datafit,
            penalty,
            all_feats.view(),
        );

        // KKT violation check
        if epoch % 10 == 0 {
            let grad_ws = construct_grad(
                X.view(),
                y.view(),
                w.view(),
                Xw.view(),
                all_feats.view(),
                datafit,
            );
            let subdiff_dist_ws =
                penalty.subdiff_distance(w.view(), grad_ws.view(), all_feats.view());
            let kkt_ws_max = get_max_arr(subdiff_dist_ws.view());
            // println!("epoch: {} :: kkt: {:#?}", epoch, kkt_ws_max);
            if kkt_ws_max < tol {
                break;
            }
        }
    }
}
