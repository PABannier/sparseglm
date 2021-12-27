extern crate ndarray;
extern crate num;

use crate::datafits::Datafit;
use crate::penalties::Penalty;
use ndarray::{s, Array1, ArrayView1, ArrayView2};
use num::Float;
use std::fmt::Debug;

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

#[rustfmt::skip]
pub fn construct_ws_from_kkt<T: Float>(kkt: &mut Array1<T>, w: ArrayView1<T>,
                                       p0:usize) -> (Array1<usize>, usize) {
    let mut non_zero_features = 0;
    let n_features = kkt.len();
    for j in 0..n_features {
        if w[j] != T::zero() {
            non_zero_features += 1;
            kkt[j] = T::infinity();
        }
    }

    let ws_size = usize::max(p0, usize::min(n_features, 2 * non_zero_features));
    let ws = Array1::<usize>::zeros(ws_size);
    // Argsort....

    (ws, ws_size)
}

#[rustfmt::skip]
pub fn construct_grad<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, w: ArrayView1<T>, Xw: ArrayView1<T>,
    ws: ArrayView1<usize>, datafit: &D) -> Array1<T> {
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
    }
    grad
}

#[rustfmt::skip]
pub fn cd_epoch<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, w: &mut Array1<T>, Xw: &mut Array1<T>,
    datafit: &D, penalty: &P, ws: ArrayView1<usize>,) {
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

#[rustfmt::skip]
pub fn solver<T: 'static + Float + Debug, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, datafit: &mut D, penalty: &P,
    w: &mut Array1<T>, Xw: &mut Array1<T>, max_iter: usize, max_epochs: usize,
    p0: usize, tol: T, verbose: bool) {
    let n_features = X.shape()[1];

    let mut kkt_max: T;
    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    datafit.initialize(X.view(), y.view());

    for iter in 0..max_iter {
        #[rustfmt::skip]
        let grad = construct_grad(
            X.view(), y.view(), w.view(), Xw.view(), all_feats.view(), datafit);
        let mut kkt = penalty.subdiff_distance(w.view(), grad.view(), all_feats.view());
        kkt_max = get_max_arr(kkt.view());

        if kkt_max <= tol {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);

        if verbose {
            println!("Iteration {} :: {} feats in subproblem.", iter + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            #[rustfmt::skip]
            cd_epoch(
                X.view(), y.view(), w, Xw, datafit, penalty, ws.view());
            // KKT violation check
            if epoch % 10 == 0 {
                #[rustfmt::skip]
                let grad_ws = construct_grad(
                    X.view(), y.view(), w.view(), Xw.view(), ws.view(), datafit);
                let kkt_ws = penalty.subdiff_distance(w.view(), grad_ws.view(), ws.view());
                let kkt_ws_max = get_max_arr(kkt_ws.view());

                if verbose {
                    println!("Epoch {} :: KKT {:#?}", epoch, kkt_ws_max);
                }
                if ws_size == n_features {
                    if kkt_ws_max < tol {
                        break;
                    }
                } else {
                    if kkt_ws_max < T::from(0.3).unwrap() * kkt_max {
                        if verbose {
                            println!("Early exit");
                        }
                        break;
                    }
                }
            }
        }
    }
}
