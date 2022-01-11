extern crate ndarray;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};

use super::Float;
use crate::datafits_multitask::MultiTaskDatafit;
use crate::datasets::DesignMatrix;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrixType, Targets};
use crate::helpers::helpers::{argsort_by, solve_lin_sys};
use crate::penalties_multitask::PenaltyMultiTask;

#[cfg(test)]
mod tests;

pub fn construct_grad<F, D, DF, DM, T>(
    dataset: &DatasetBase<DM, T>,
    XW: ArrayView2<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
) -> Array2<F>
where
    F: 'static + Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: MultiTaskDatafit<F, D, DM, T>,
{
    let ws_size = ws.len();
    let n_tasks = dataset.n_tasks();
    let mut grad = Array2::<F>::zeros((ws_size, n_tasks));
    for (idx, &j) in ws.iter().enumerate() {
        let grad_j = datafit.gradient_j(&dataset, XW, j);
        for t in 0..n_tasks {
            grad[[idx, t]] = grad_j[t];
        }
    }
    grad
}

pub fn kkt_violation<F, D, DF, P, DM, T>(
    dataset: &DatasetBase<DM, T>,
    W: ArrayView2<F>,
    XW: ArrayView2<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
    penalty: &P,
) -> (Array1<F>, F)
where
    F: 'static + Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: MultiTaskDatafit<F, D, DM, T>,
    P: PenaltyMultiTask<F, D>,
{
    let grad_ws = construct_grad(&dataset, XW, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(W, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

pub fn construct_ws_from_kkt<F>(
    kkt: &mut Array1<F>,
    W: ArrayView2<F>,
    p0: usize,
) -> (Array1<usize>, usize)
where
    F: 'static + Float,
{
    let n_features = W.shape()[0];
    let mut nnz_features: usize = 0;

    for j in 0..n_features {
        if W.slice(s![j, ..]).map(|&x| x.abs()).sum() != F::zero() {
            nnz_features += 1;
            kkt[j] = F::infinity();
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

pub fn anderson_accel<F, D, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    W: &mut Array2<F>,
    XW: &mut Array2<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
    last_K_W: &mut Array2<F>,
    U: &mut Array2<F>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    F: 'static + Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: MultiTaskDatafit<F, D, DM, T>,
    P: PenaltyMultiTask<F, D>,
{
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();
    let n_tasks = dataset.n_tasks();

    let X = dataset.design_matrix;
    let Y = dataset.targets;

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

        let mut C: Array2<F> = Array2::zeros((K, K));
        // general_mat_mul is 20x slower than using plain for loops
        // Complexity relatively low o(K^2 * ws_size) considering K usually is 5
        for i in 0..K {
            for j in 0..K {
                for l in 0..ws.len() {
                    C[[i, j]] += U[[i, l]] * U[[j, l]];
                }
            }
        }

        let _res = solve_lin_sys(C.view(), Array1::<F>::ones(K).view());

        match _res {
            Ok(z) => {
                let denom = z.sum();
                let c = z.map(|&x| x / denom);

                let mut W_acc = Array2::<F>::zeros((n_features, n_tasks));

                // Extrapolation
                for (idx, &j) in ws.iter().enumerate() {
                    for k in 0..K {
                        for t in 0..n_tasks {
                            W_acc[[j, t]] += last_K_W[[k, idx * n_tasks + t]] * c[k];
                        }
                    }
                }

                let mut XW_acc = Array2::<F>::zeros((n_samples, n_tasks));

                match dataset.matrix_type() {
                    DesignMatrixType::Dense => {
                        for i in 0..n_samples {
                            for &j in ws {
                                for t in 0..n_tasks {
                                    XW_acc[[i, t]] += X[[i, j]] * W_acc[[j, t]];
                                }
                            }
                        }
                    }
                    DesignMatrixType::Sparse => {
                        for &j in ws {
                            for idx in X.indptr[j]..X.indptr[j + 1] {
                                for t in 0..n_tasks {
                                    XW_acc[[X.indices[idx as usize] as usize, t]] +=
                                        X.data[idx as usize] * W_acc[[j, t]];
                                }
                            }
                        }
                    }
                }

                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());
                let p_obj_acc = datafit.value(dataset, XW_acc.view()) + penalty.value(W_acc.view());

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

pub fn bcd_epoch<F, D, DF, P, T>(
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    W: &mut Array2<F>,
    XW: &mut Array2<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
) where
    F: 'static + Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
    DF: MultiTaskDatafit<F, D, ArrayBase<D, Ix2>, T>,
    P: PenaltyMultiTask<F, D>,
{
    let n_samples = dataset.n_samples();
    let n_tasks = dataset.n_tasks();

    let X = dataset.design_matrix;
    let lipschitz = datafit.lipschitz();

    for &j in ws {
        if lipschitz[j] == F::zero() {
            continue;
        }
        let Xj: ArrayView1<F> = X.slice(s![.., j]);
        let mut old_W_j = Array1::<F>::zeros(n_tasks);
        for t in 0..n_tasks {
            old_W_j[t] = W[[j, t]];
        }
        let grad_j = datafit.gradient_j(dataset, XW.view(), j);

        let mut upd = Array1::<F>::zeros(n_tasks);
        for t in 0..n_tasks {
            upd[t] = old_W_j[t] - grad_j[t] / lipschitz[j];
        }
        let upd = penalty.prox_op(upd.view(), F::one() / lipschitz[j]);
        for t in 0..n_tasks {
            W[[j, t]] = upd[t];
        }

        let mut diff = Array1::<F>::zeros(n_tasks);
        let mut sum_diff = F::zero();
        for t in 0..n_tasks {
            diff[t] = W[[j, t]] - old_W_j[t];
            sum_diff += diff[t].abs()
        }

        if sum_diff != F::zero() {
            for i in 0..n_samples {
                for t in 0..n_tasks {
                    XW[[i, t]] += diff[t] * Xj[i];
                }
            }
        }
    }
}

pub fn bcd_epoch_sparse<'a, F, D, DF, P, T>(
    dataset: &DatasetBase<CSCArray<'a, F>, T>,
    W: &mut Array2<F>,
    XW: &mut Array2<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
) where
    F: 'static + Float,
    D: Data<Elem = F>,
    DF: MultiTaskDatafit<F, D, CSCArray<'a, F>, T>,
    P: PenaltyMultiTask<F, D>,
    T: Targets<Elem = F>,
{
    let n_tasks = dataset.n_tasks();

    let X = dataset.design_matrix;
    let lipschitz = datafit.lipschitz();

    for &j in ws {
        if lipschitz[j] == F::zero() {
            continue;
        }
        let mut old_W_j = Array1::<F>::zeros(n_tasks);
        for t in 0..n_tasks {
            old_W_j[t] = W[[j, t]];
        }
        let grad_j = datafit.gradient_j(dataset, XW.view(), j);

        let mut upd = Array1::<F>::zeros(n_tasks);
        for t in 0..n_tasks {
            upd[t] = old_W_j[t] - grad_j[t] / lipschitz[j];
        }
        let upd = penalty.prox_op(upd.view(), F::one() / lipschitz[j]);
        for t in 0..n_tasks {
            W[[j, t]] = upd[t];
        }

        let mut diff = Array1::<F>::zeros(n_tasks);
        let mut sum_diff = F::zero();
        for t in 0..n_tasks {
            diff[t] = W[[j, t]] - old_W_j[t];
            sum_diff += diff[t].abs();
        }

        if sum_diff != F::zero() {
            for i in X.indptr[j]..X.indptr[j + 1] {
                for t in 0..n_tasks {
                    XW[[X.indices[i as usize] as usize, t]] += diff[t] * X.data[i as usize];
                }
            }
        }
    }
}

pub fn solver_multitask<F, D, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    datafit: &mut DF,
    penalty: &P,
    max_iter: usize,
    max_epochs: usize,
    _p0: usize,
    tol: F,
    use_accel: bool,
    K: usize,
    verbose: bool,
) -> Array2<F>
where
    F: 'static + Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: MultiTaskDatafit<F, D, DM, T>,
    P: PenaltyMultiTask<F, D>,
{
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();
    let n_tasks = dataset.n_tasks();

    datafit.initialize(dataset);

    let X = dataset.design_matrix;
    let Y = dataset.targets;

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if _p0 > n_features { n_features } else { _p0 };

    let mut W = Array2::<F>::zeros((n_features, n_tasks));
    let mut XW = Array2::<F>::zeros((n_samples, n_tasks));

    for t in 0..max_iter {
        let (mut kkt, kkt_max) = kkt_violation(
            dataset,
            W.view(),
            XW.view(),
            all_feats.view(),
            datafit,
            penalty,
        );

        if verbose {
            println!("KKT max violation: {:#?}", kkt_max);
        }
        if kkt_max <= tol {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, W.view(), p0);

        let mut last_K_W = Array2::<F>::zeros((K + 1, ws_size * n_tasks));
        let mut U = Array2::<F>::zeros((K, ws_size * n_tasks));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            match dataset.matrix_type() {
                DesignMatrixType::Dense => {
                    bcd_epoch(dataset, &mut W, &mut XW, datafit, penalty, ws.view());
                }
                DesignMatrixType::Sparse => {
                    bcd_epoch_sparse(dataset, &mut W, &mut XW, datafit, penalty, ws.view());
                }
            }

            // Anderson acceleration
            if use_accel {
                anderson_accel(
                    dataset,
                    &mut W,
                    &mut XW,
                    datafit,
                    penalty,
                    ws.view(),
                    &mut last_K_W,
                    &mut U,
                    epoch,
                    K,
                    verbose,
                );
            }

            // KKT violation check
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());

                let (_, kkt_ws_max) =
                    kkt_violation(dataset, W.view(), XW.view(), ws.view(), datafit, penalty);

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
                    if kkt_ws_max < F::cast(0.3) * kkt_max {
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
