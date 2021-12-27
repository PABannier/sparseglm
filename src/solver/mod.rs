extern crate ndarray;
extern crate num;

use crate::datafits::Datafit;
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

pub fn construct_grad<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    w: ArrayView1<T>,
    Xw: ArrayView1<T>,
    Xty: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: D,
) -> Array1<T> {
    let n_samples = X.shape()[0];
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        grad[idx] = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
    }
    grad
}

pub fn cd_epoch<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    w: &mut Array1<T>,
    datafit: D,
    lipschitz: ArrayView1<T>,
    Xty: ArrayView1<T>,
    Xw: &mut Array1<T>,
    ws: ArrayView1<usize>,
    alpha: T,
) {
    let n_samples = X.shape()[0];

    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        let old_w_j = w[j];
        let grad_j = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
        w[j] = soft_thresholding(old_w_j - grad_j / lipschitz[j], alpha / lipschitz[j]);
        if w[j] != old_w_j {
            for i in 0..n_samples {
                Xw[i] = Xw[i] + (w[j] - old_w_j) * Xj[i];
            }
        }
    }
}

pub fn solver<T: 'static + Float>(
    X: ArrayView2<T>,
    y: ArrayView1<T>,
    alpha: T,
    max_iter: usize,
    max_epochs: usize,
    kkt_check_iter: usize,
    p0: usize,
    tol: T,
) -> Array1<T> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];

    let mut kkt_max: T;
    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let Xty = X.t().dot(&y);

    let lipschitz = compute_lipschitz(X.view(), n_samples);
    let mut w = Array1::<T>::zeros(n_features);
    let mut xw = Array1::<T>::zeros(n_samples);

    for iter in 0..max_iter {
        let grad = construct_grad(X.view(), xw.view(), Xty.view(), all_feats.view());
        let mut kkt = compute_dist_to_subdiff(grad.view(), w.view(), alpha, all_feats.view());
        kkt_max = get_max_arr(kkt.view());

        if kkt_max <= tol {
            break;
        }

        let mut non_zero_features = 0;
        for j in 0..n_features {
            if w[j] != T::zero() {
                non_zero_features += 1;
                kkt[j] = T::infinity();
            }
        }

        let ws_size = usize::max(p0, usize::min(n_features, 2 * non_zero_features));
        let ws: Array1<usize> = get_ws_from_kkt(kkt.view(), ws_size);

        println!("Iteration {}, {} feats in subproblem.", iter + 1, ws_size);

        for epoch in 0..max_epochs {
            cd_epoch(
                X.view(),
                &mut w,
                lipschitz.view(),
                Xty.view(),
                &mut xw,
                ws.view(),
                alpha,
            );

            // KKT violation check
            if epoch % kkt_check_iter == 0 {
                let grad_ws = construct_grad(X.view(), xw.view(), Xty.view(), ws.view());
                let subdiff_dist_ws =
                    compute_dist_to_subdiff(grad_ws.view(), w.view(), alpha, ws.view());
                let kkt_ws_max = get_max_arr(subdiff_dist_ws.view());
                println!("epoch: {} :: kkt: {:#?}", epoch, kkt_ws_max);
                if ws_size == n_features {
                    if kkt_ws_max < tol {
                        break;
                    }
                } else {
                    if kkt_ws_max < T::from(0.3).unwrap() * kkt_max {
                        println!("Early exit");
                        break;
                    }
                }
            }
        }
    }

    w
}
