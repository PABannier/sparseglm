extern crate ndarray;
extern crate ndarray_stats;
extern crate num;

use ndarray::{s, Array1, ArrayView1, ArrayView2};
use num::Float;
use std::fmt::Debug;

use crate::datafits::Datafit;
use crate::penalties::Penalty;

#[cfg(test)]
mod tests;

pub fn soft_thresholding<T: Float>(x: T, threshold: T) -> T {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        T::zero()
    }
}

#[rustfmt::skip]
pub fn construct_grad<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, w: ArrayView1<T>, Xw: ArrayView1<T>,
    ws: &[usize], datafit: &D) -> Array1<T> {
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_scalar(X.view(), y.view(), w.view(), Xw.view(), j);
    }
    grad
}

#[rustfmt::skip]
pub fn kkt_violation<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, w: ArrayView1<T>, Xw: ArrayView1<T>,
    ws: &[usize], datafit: &D, penalty: &P) -> (Vec<T>, T) {
    let grad_ws = construct_grad(X.view(), y.view(), w.view(), Xw.view(), &ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w.view(), grad_ws.view(), &ws);
    (kkt_ws, kkt_ws_max)
}

#[rustfmt::skip]
pub fn construct_ws_from_kkt<T: 'static + Float>(
    kkt: &mut Vec<T>, w: ArrayView1<T>, p0: usize) -> (Vec<usize>, usize){
    let n_features = w.len();
    let mut nnz_features: usize = 0;

    for j in 0..n_features {
        if w[j] != T::zero() {
            nnz_features += 1;
            kkt[j] = T::infinity();
        }
    }
    // TODO: Correction for p0 (handling the case p0 > n_features)
    // let p0 = usize::min(p0, usize::max(nnz_features, 1));
    let ws_size = usize::max(p0, usize::min(2 * nnz_features, n_features));

    let mut kkt_with_indices: Vec<(usize, T)> = kkt
        .iter()
        .copied()
        .enumerate()
        .collect();
    kkt_with_indices.sort_unstable_by(|(_, p), (_, q)| {
        // Swapped order for sorting in descending order.
        q.partial_cmp(p).expect("kkt must not be NaN.")
    });
    let ws: Vec<usize> = kkt_with_indices
        .iter()
        .map(|&(ind, _)| ind)
        .take(ws_size)
        .collect();
    (ws, ws_size)
}

#[rustfmt::skip]
pub fn cd_epoch<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>, y: ArrayView1<T>, w: &mut Array1<T>, Xw: &mut Array1<T>,
    datafit: &D, penalty: &P, ws: &[usize]) {
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
    max_iter: usize, max_epochs: usize, p0: usize, tol: T, verbose: bool) 
    -> Array1<T> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];
    let all_feats: Vec<usize> = (0..n_features).collect();

    datafit.initialize(X.view(), y.view());

    let mut w = Array1::<T>::zeros(n_features);
    let mut Xw = Array1::<T>::zeros(n_samples);

    for t in 0..max_iter {
        #[rustfmt::skip]
        let (mut kkt, kkt_max) = kkt_violation(
            X.view(), y.view(), w.view(), Xw.view(), &all_feats, datafit, penalty);

        if verbose {
            println!("KKT max violation: {:#?}", kkt_max);
        }
        if kkt_max <= tol {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);
        if verbose{
            println!("Iteration {}, {} features in subproblem.", t+1, ws_size);
        }

        for epoch in 0..max_epochs {
            #[rustfmt::skip]
            cd_epoch(
                X.view(), y.view(), &mut w, &mut Xw, datafit, penalty, &ws);
    
            // KKT violation check
            if epoch % 10 == 0 {
                let p_obj = datafit.value(y.view(), w.view(), Xw.view()) + penalty.value(w.view());
                #[rustfmt::skip]
                let (_, kkt_ws_max) = kkt_violation(
                    X.view(), y.view(), w.view(), Xw.view(), &ws, datafit,
                    penalty);

                if verbose {
                    println!(
                        "epoch: {} :: obj: {:#?} :: kkt: {:#?}",
                        epoch, p_obj, kkt_ws_max
                    );
                }

                if ws_size == n_features {
                    if kkt_ws_max <= tol {
                        break;
                    }
                } else {
                    if kkt_ws_max < T::from(0.3).unwrap() * kkt_max {
                        if verbose {
                            println!("Early exit.")
                        }
                        break;
                    }
                }
                
                
            }
        }

    }

    w
}
