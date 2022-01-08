extern crate ndarray;
extern crate num;

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use num::Float;
use std::fmt::Debug;

use crate::datafits_multitask::DatafitMultiTask;
use crate::helpers::helpers::solve_lin_sys;
use crate::penalties_multitask::PenaltyMultiTask;
use crate::sparse::{CSCArray, MatrixParam};

#[cfg(test)]
mod tests;

pub fn construct_grad<T: 'static + Float, D: DatafitMultiTask<T>>(
    X: ArrayView2<T>,
    XW: ArrayView2<T>,
    ws: &[usize],
    datafit: &D,
) -> Array2<T> {
    let ws_size = ws.len();
    let n_tasks = XW.shape()[1];
    let mut grad = Array2::<T>::zeros((ws_size, n_tasks));
    for (idx, &j) in ws.iter().enumerate() {
        let grad_j = datafit.gradient_j(X, XW.view(), j);
        for t in 0..n_tasks {
            grad[[idx, t]] = grad_j[t];
        }
    }
    grad
}

pub fn construct_grad_sparse<T: 'static + Float, D: DatafitMultiTask<T>>(
    X: &CSCArray<T>,
    XW: ArrayView2<T>,
    ws: &[usize],
    datafit: &D,
) -> Array2<T> {
    let ws_size = ws.len();
    let n_tasks = XW.shape()[1];
    let mut grad = Array2::<T>::zeros((ws_size, n_tasks));
    for (idx, &j) in ws.iter().enumerate() {
        let grad_j = datafit.gradient_j_sparse(&X, XW.view(), j);
        for t in 0..n_tasks {
            grad[[idx, t]] = grad_j[t];
        }
    }
    grad
}

pub fn kkt_violation<T: 'static + Float, D: DatafitMultiTask<T>, P: PenaltyMultiTask<T>>(
    X: ArrayView2<T>,
    W: ArrayView2<T>,
    XW: ArrayView2<T>,
    ws: &[usize],
    datafit: &D,
    penalty: &P,
) -> (Vec<T>, T) {
    let grad_ws = construct_grad(X, XW.view(), &ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(W.view(), grad_ws.view(), &ws);
    (kkt_ws, kkt_ws_max)
}

pub fn kkt_violation_sparse<T: 'static + Float, D: DatafitMultiTask<T>, P: PenaltyMultiTask<T>>(
    X: &CSCArray<T>,
    W: ArrayView2<T>,
    XW: ArrayView2<T>,
    ws: &[usize],
    datafit: &D,
    penalty: &P,
) -> (Vec<T>, T) {
    let grad_ws = construct_grad_sparse(&X, XW.view(), &ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(W.view(), grad_ws.view(), &ws);
    (kkt_ws, kkt_ws_max)
}

pub fn construct_ws_from_kkt<T: 'static + Float>(
    kkt: &mut Vec<T>,
    W: ArrayView2<T>,
    p0: usize,
) -> (Vec<usize>, usize) {
    let n_features = W.shape()[0];
    let mut nnz_features: usize = 0;

    for j in 0..n_features {
        if W.slice(s![j, ..]).map(|&x| x.abs()).sum() != T::zero() {
            nnz_features += 1;
            kkt[j] = T::infinity();
        }
    }

    let ws_size = usize::max(p0, usize::min(2 * nnz_features, n_features));

    let mut kkt_with_indices: Vec<(usize, T)> = kkt.iter().copied().enumerate().collect();
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

pub fn anderson_accel<T, D, P>(
    Y: ArrayView2<T>,
    X: MatrixParam<T>,
    W: &mut Array2<T>,
    XW: &mut Array2<T>,
    datafit: &D,
    penalty: &P,
    ws: &[usize],
    last_K_W: &mut Array2<T>,
    U: &mut Array2<T>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    T: 'static + Float + Debug,
    D: DatafitMultiTask<T>,
    P: PenaltyMultiTask<T>,
{
    let n_samples = Y.shape()[0];
    let n_tasks = Y.shape()[1];
    let n_features = W.shape()[0];
    // last_K_w[epoch % (K + 1)] = w[ws]
    for (idx, &j) in ws.iter().enumerate() {
        for t in 0..n_tasks {
            last_K_W[[epoch % (K + 1), idx * n_tasks + t]] = W[[j, t]];
        }
    }

    if epoch % (K + 1) == K {
        for k in 0..K {
            for j in 0..(ws.len() * n_tasks) {
                U[[k, j]] = last_K_W[[k + 1, j]] - last_K_W[[k, j]];
            }
        }

        let mut C: Array2<T> = Array2::zeros((K, K));
        general_mat_mul(T::one(), &U, &U.t(), T::one(), &mut C);

        let _res = solve_lin_sys(C.view(), Array1::<T>::ones(K).view());

        match _res {
            Ok(z) => {
                let denom = z.sum();
                let c = z.map(|&x| x / denom);

                let mut W_acc = Array2::<T>::zeros((n_features, n_tasks));

                // Extrapolation
                for (idx, &j) in ws.iter().enumerate() {
                    for k in 0..K {
                        for t in 0..n_tasks {
                            W_acc[[j, t]] = W_acc[[j, t]] + last_K_W[[k, idx * n_tasks + t]] * c[k];
                        }
                    }
                }

                let mut XW_acc = Array2::<T>::zeros((n_samples, n_tasks));

                match X {
                    MatrixParam::DenseMatrix(X_full) => {
                        for i in 0..n_samples {
                            for &j in ws {
                                for t in 0..n_tasks {
                                    XW_acc[[i, t]] =
                                        XW_acc[[i, t]] + X_full[[i, j]] * W_acc[[j, t]];
                                }
                            }
                        }
                    }
                    MatrixParam::SparseMatrix(_) => {
                        // TODO: Implement with sparse matrices
                        // Xw_acc += X_full
                    }
                }

                let p_obj = datafit.value(Y, XW.view()) + penalty.value(W.view());
                let p_obj_acc = datafit.value(Y, XW_acc.view()) + penalty.value(W_acc.view());

                if p_obj_acc < p_obj {
                    W.assign(&W_acc);
                    XW.assign(&XW_acc);

                    if verbose {
                        println!("[ACCEL] p_obj {:#?} :: p_obj_acc {:#?}", p_obj, p_obj_acc);
                    }
                }
            }
            Err(_) => {
                if verbose {
                    println!("----LinAlg error");
                }
            }
        }
    }
}

pub fn bcd_epoch<T: 'static + Float, D: DatafitMultiTask<T>, P: PenaltyMultiTask<T>>(
    X: ArrayView2<T>,
    W: &mut Array2<T>,
    XW: &mut Array2<T>,
    datafit: &D,
    penalty: &P,
    ws: &[usize],
) {
    let n_samples = X.shape()[0];
    let n_tasks = W.shape()[1];
    let lipschitz = datafit.get_lipschitz();
    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        let old_W_j = W.slice(s![j, ..]);
        let grad_j = datafit.gradient_j(X, XW.view(), j);

        let mut upd = Array1::<T>::zeros(n_tasks);
        for t in 0..n_tasks {
            upd[t] = old_W_j[t] - grad_j[t] / lipschitz[j];
        }
        let upd = penalty.prox_op(upd.view(), T::one() / lipschitz[j]);
        for t in 0..n_tasks {
            W[[j, t]] = upd[t];
        }

        let mut diff = Array1::<T>::zeros(n_tasks);
        let mut sum_diff = T::zero();
        for t in 0..n_tasks {
            diff[t] = W[[j, t]] - old_W_j[t];
            sum_diff = sum_diff + diff[t].abs()
        }

        if sum_diff != T::zero() {
            for i in 0..n_samples {
                for t in 0..n_tasks {
                    XW[[i, t]] = XW[[i, t]] + diff[t] * Xj[i];
                }
            }
        }
    }
}

pub fn bcd_epoch_sparse<T: 'static + Float, D: DatafitMultiTask<T>, P: PenaltyMultiTask<T>>(
    X: &CSCArray<T>,
    W: &mut Array2<T>,
    XW: &mut Array2<T>,
    datafit: &D,
    penalty: &P,
    ws: &[usize],
) {
    let n_tasks = W.shape()[1];
    let lipschitz = datafit.get_lipschitz();
    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let old_W_j = W.slice(s![j, ..]);
        let grad_j = datafit.gradient_j_sparse(&X, XW.view(), j);

        let mut upd = Array1::<T>::zeros(n_tasks);
        for t in 0..n_tasks {
            upd[t] = old_W_j[t] - grad_j[t] / lipschitz[j];
        }
        let upd = penalty.prox_op(upd.view(), T::one() / lipschitz[j]);
        for t in 0..n_tasks {
            W[[j, t]] = upd[t];
        }

        let mut diff = Array1::<T>::zeros(n_tasks);
        let mut sum_diff = T::zero();
        for t in 0..n_tasks {
            diff[t] = W[[j, t]] - old_W_j[t];
            sum_diff = sum_diff + diff[t].abs();
        }

        if sum_diff != T::zero() {
            for i in X.indptr[j]..X.indptr[j + 1] {
                for t in 0..n_tasks {
                    XW[[X.indices[i], t]] = XW[[X.indices[i], t]] + diff[t] * X.data[i];
                }
            }
        }
    }
}

pub fn solver_multitask<
    T: 'static + Float + Debug,
    D: DatafitMultiTask<T>,
    P: PenaltyMultiTask<T>,
>(
    X: MatrixParam<T>,
    Y: ArrayView2<T>,
    datafit: &mut D,
    penalty: &P,
    max_iter: usize,
    max_epochs: usize,
    _p0: usize,
    tol: T,
    use_accel: bool,
    K: usize,
    verbose: bool,
) -> Array2<T> {
    let n_samples = Y.shape()[0];
    let n_tasks = Y.shape()[1];
    let n_features: usize;

    match X {
        MatrixParam::DenseMatrix(X_full) => {
            datafit.initialize(X_full.view(), Y.view());
            n_features = X_full.shape()[1];
        }
        MatrixParam::SparseMatrix(X_sparse) => {
            datafit.initialize_sparse(X_sparse, Y.view());
            n_features = X_sparse.indptr.len() - 1;
        }
    }

    let all_feats: Vec<usize> = (0..n_features).collect();

    let p0 = if _p0 > n_features { n_features } else { _p0 };

    let mut W = Array2::<T>::zeros((n_features, n_tasks));
    let mut XW = Array2::<T>::zeros((n_samples, n_tasks));

    for t in 0..max_iter {
        let mut kkt: Vec<T>;
        let kkt_max: T;

        match X {
            MatrixParam::DenseMatrix(X_full) => {
                let (a, b) =
                    kkt_violation(X_full, W.view(), XW.view(), &all_feats, datafit, penalty);
                kkt = a;
                kkt_max = b;
            }
            MatrixParam::SparseMatrix(X_sparse) => {
                let (a, b) = kkt_violation_sparse(
                    X_sparse,
                    W.view(),
                    XW.view(),
                    &all_feats,
                    datafit,
                    penalty,
                );
                kkt = a;
                kkt_max = b;
            }
        }

        if verbose {
            println!("KKT max violation: {:#?}", kkt_max);
        }
        if kkt_max <= tol {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, W.view(), p0);

        let mut last_K_W = Array2::<T>::zeros((K + 1, ws_size * n_tasks));
        let mut U = Array2::<T>::zeros((K, ws_size * n_tasks));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            match X {
                MatrixParam::DenseMatrix(X_full) => {
                    bcd_epoch(X_full.view(), &mut W, &mut XW, datafit, penalty, &ws);
                }
                MatrixParam::SparseMatrix(X_sparse) => {
                    bcd_epoch_sparse(X_sparse, &mut W, &mut XW, datafit, penalty, &ws);
                }
            }

            // Anderson acceleration
            if use_accel {
                anderson_accel(
                    Y,
                    X,
                    &mut W,
                    &mut XW,
                    datafit,
                    penalty,
                    &ws,
                    &mut last_K_W,
                    &mut U,
                    epoch,
                    K,
                    verbose,
                );
            }

            // KKT violation check
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(Y, XW.view()) + penalty.value(W.view());

                let kkt_ws_max: T;

                match X {
                    MatrixParam::DenseMatrix(X_full) => {
                        let (_, b) =
                            kkt_violation(X_full, W.view(), XW.view(), &ws, datafit, penalty);
                        kkt_ws_max = b;
                    }
                    MatrixParam::SparseMatrix(X_sparse) => {
                        let (_, b) = kkt_violation_sparse(
                            X_sparse,
                            W.view(),
                            XW.view(),
                            &ws,
                            datafit,
                            penalty,
                        );
                        kkt_ws_max = b;
                    }
                }

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

    W
}
