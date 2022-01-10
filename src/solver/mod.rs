extern crate ndarray;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use super::Float;
use crate::datafits::Datafit;
use crate::helpers::helpers::{argsort_by, solve_lin_sys};
use crate::penalties::Penalty;
use crate::sparse::{CSCArray, MatrixParam};

#[cfg(test)]
mod tests;

pub fn construct_grad<T: 'static + Float, D: Datafit<T>>(
    X: ArrayView2<T>,
    Xw: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: &D,
) -> Array1<T> {
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_j(X, Xw, j);
    }
    grad
}

pub fn construct_grad_sparse<T: 'static + Float, D: Datafit<T>>(
    X: &CSCArray<T>,
    Xw: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: &D,
) -> Array1<T> {
    let ws_size = ws.len();
    let mut grad = Array1::<T>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_j_sparse(&X, Xw, j);
    }
    grad
}

pub fn kkt_violation<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>,
    w: ArrayView1<T>,
    Xw: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: &D,
    penalty: &P,
) -> (Array1<T>, T) {
    let grad_ws = construct_grad(X, Xw, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

pub fn kkt_violation_sparse<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: &CSCArray<T>,
    w: ArrayView1<T>,
    Xw: ArrayView1<T>,
    ws: ArrayView1<usize>,
    datafit: &D,
    penalty: &P,
) -> (Array1<T>, T) {
    let grad_ws = construct_grad_sparse(&X, Xw, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

pub fn construct_ws_from_kkt<T: 'static + Float>(
    kkt: &mut Array1<T>,
    w: ArrayView1<T>,
    p0: usize,
) -> (Array1<usize>, usize) {
    let n_features = w.len();
    let mut nnz_features: usize = 0;

    for j in 0..n_features {
        if w[j] != T::zero() {
            nnz_features += 1;
            kkt[j] = T::infinity();
        }
    }
    let ws_size = usize::max(p0, usize::min(2 * nnz_features, n_features));

    let mut sorted_indices = argsort_by(&kkt, |a, b| {
        // Swapped order for sorting in descending order
        b.partial_cmp(a).expect("Elements must not be NaN.")
    });
    sorted_indices.truncate(ws_size);

    let ws = Array1::from_shape_vec(ws_size, sorted_indices).unwrap();
    (ws, ws_size)
}

pub fn anderson_accel<T, D, P>(
    y: ArrayView1<T>,
    X: MatrixParam<T>,
    w: &mut Array1<T>,
    Xw: &mut Array1<T>,
    datafit: &D,
    penalty: &P,
    ws: ArrayView1<usize>,
    last_K_w: &mut Array2<T>,
    U: &mut Array2<T>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    T: 'static + Float,
    D: Datafit<T>,
    P: Penalty<T>,
{
    // last_K_w[epoch % (K + 1)] = w[ws]
    // Note: from my experiments, loops are 4-5x faster than slice
    // See: https://github.com/rust-ndarray/ndarray/issues/571
    for (idx, &j) in ws.iter().enumerate() {
        last_K_w[[epoch % (K + 1), idx]] = w[j];
    }

    if epoch % (K + 1) == K {
        for k in 0..K {
            for j in 0..ws.len() {
                U[[k, j]] = last_K_w[[k + 1, j]] - last_K_w[[k, j]];
            }
        }

        let mut C: Array2<T> = Array2::zeros((K, K));
        // general_mat_mul is 20x slower than using plain for loops
        // Complexity relatively low o(K^2 * ws_size) considering K usually is 5
        for i in 0..K {
            for j in 0..K {
                for l in 0..ws.len() {
                    C[[i, j]] = C[[i, j]] + U[[i, l]] * U[[j, l]];
                }
            }
        }

        let _res = solve_lin_sys(C.view(), Array1::<T>::ones(K).view());

        match _res {
            Ok(z) => {
                let denom = z.sum();
                let c = z.map(|&x| x / denom);

                let mut w_acc = Array1::<T>::zeros(w.len());

                // Extrapolation
                for (idx, &j) in ws.iter().enumerate() {
                    for k in 0..K {
                        w_acc[j] = w_acc[j] + last_K_w[[k, idx]] * c[k];
                    }
                }

                let mut Xw_acc = Array1::<T>::zeros(y.len());

                match X {
                    MatrixParam::DenseMatrix(X_full) => {
                        for i in 0..Xw_acc.len() {
                            for &j in ws {
                                Xw_acc[i] = Xw_acc[i] + X_full[[i, j]] * w_acc[j];
                            }
                        }
                    }
                    MatrixParam::SparseMatrix(X_sparse) => {
                        for &j in ws {
                            for idx in X_sparse.indptr[j]..X_sparse.indptr[j + 1] {
                                Xw_acc[X_sparse.indices[idx as usize] as usize] = Xw_acc
                                    [X_sparse.indices[idx as usize] as usize]
                                    + X_sparse.data[idx as usize] * w_acc[j];
                            }
                        }
                    }
                }

                let p_obj = datafit.value(y, Xw.view()) + penalty.value(w.view());
                let p_obj_acc = datafit.value(y, Xw_acc.view()) + penalty.value(w_acc.view());

                if p_obj_acc < p_obj {
                    w.assign(&w_acc);
                    Xw.assign(&Xw_acc);

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

pub fn cd_epoch<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: ArrayView2<T>,
    w: &mut Array1<T>,
    Xw: &mut Array1<T>,
    datafit: &D,
    penalty: &P,
    ws: ArrayView1<usize>,
) {
    let n_samples = X.shape()[0];
    let lipschitz = datafit.lipschitz();

    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let old_w_j = w[j];
        let grad_j = datafit.gradient_j(X, Xw.view(), j);
        w[j] = penalty.prox_op(old_w_j - grad_j / lipschitz[j], T::one() / lipschitz[j]);
        if w[j] != old_w_j {
            for i in 0..n_samples {
                Xw[i] = Xw[i] + (w[j] - old_w_j) * X[[i, j]];
            }
        }
    }
}

pub fn cd_epoch_sparse<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: &CSCArray<T>,
    w: &mut Array1<T>,
    Xw: &mut Array1<T>,
    datafit: &D,
    penalty: &P,
    ws: ArrayView1<usize>,
) {
    let lipschitz = datafit.lipschitz();

    for &j in ws {
        if lipschitz[j] == T::zero() {
            continue;
        }
        let old_w_j = w[j];
        let grad_j = datafit.gradient_j_sparse(&X, Xw.view(), j);
        w[j] = penalty.prox_op(old_w_j - grad_j / lipschitz[j], T::one() / lipschitz[j]);
        let diff = w[j] - old_w_j;
        if diff != T::zero() {
            for i in X.indptr[j]..X.indptr[j + 1] {
                Xw[X.indices[i as usize] as usize] =
                    Xw[X.indices[i as usize] as usize] + diff * X.data[i as usize];
            }
        }
    }
}

pub fn solver<T: 'static + Float, D: Datafit<T>, P: Penalty<T>>(
    X: MatrixParam<T>,
    y: ArrayView1<T>,
    datafit: &mut D,
    penalty: &P,
    max_iter: usize,
    max_epochs: usize,
    _p0: usize,
    tol: T,
    use_accel: bool,
    K: usize,
    verbose: bool,
) -> Array1<T> {
    let n_samples = y.len();
    let n_features: usize;

    match X {
        MatrixParam::DenseMatrix(X_full) => {
            datafit.initialize(X_full, y);
            n_features = X_full.shape()[1];
        }
        MatrixParam::SparseMatrix(X_sparse) => {
            datafit.initialize_sparse(X_sparse, y);
            n_features = X_sparse.indptr.len() - 1;
        }
    }

    let all_feats: Array1<usize> =
        Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if _p0 > n_features { n_features } else { _p0 };

    let mut w = Array1::<T>::zeros(n_features);
    let mut Xw = Array1::<T>::zeros(n_samples);

    for t in 0..max_iter {
        let mut kkt;
        let kkt_max: T;

        match X {
            MatrixParam::DenseMatrix(X_full) => {
                let (a, b) = kkt_violation(
                    X_full,
                    w.view(),
                    Xw.view(),
                    all_feats.view(),
                    datafit,
                    penalty,
                );
                kkt = a;
                kkt_max = b;
            }
            MatrixParam::SparseMatrix(X_sparse) => {
                let (a, b) = kkt_violation_sparse(
                    X_sparse,
                    w.view(),
                    Xw.view(),
                    all_feats.view(),
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

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);

        let mut last_K_w = Array2::<T>::zeros((K + 1, ws_size));
        let mut U = Array2::<T>::zeros((K, ws_size));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            match X {
                MatrixParam::DenseMatrix(X_full) => {
                    cd_epoch(X_full, &mut w, &mut Xw, datafit, penalty, ws.view());
                }
                MatrixParam::SparseMatrix(X_sparse) => {
                    cd_epoch_sparse(X_sparse, &mut w, &mut Xw, datafit, penalty, ws.view());
                }
            }

            // Anderson acceleration
            if use_accel {
                anderson_accel(
                    y,
                    X,
                    &mut w,
                    &mut Xw,
                    datafit,
                    penalty,
                    ws.view(),
                    &mut last_K_w,
                    &mut U,
                    epoch,
                    K,
                    verbose,
                );
            }

            // KKT violation check
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(y, Xw.view()) + penalty.value(w.view());

                let kkt_ws_max: T;

                match X {
                    MatrixParam::DenseMatrix(X_full) => {
                        let (_, b) =
                            kkt_violation(X_full, w.view(), Xw.view(), ws.view(), datafit, penalty);
                        kkt_ws_max = b;
                    }
                    MatrixParam::SparseMatrix(X_sparse) => {
                        let (_, b) = kkt_violation_sparse(
                            X_sparse,
                            w.view(),
                            Xw.view(),
                            ws.view(),
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

    w
}
