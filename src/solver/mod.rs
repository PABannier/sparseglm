extern crate ndarray;

use ndarray::{Array1, Array2, ArrayBase, ArrayView1, Data, Ix2};

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, DesignMatrixType, Targets};
use crate::helpers::helpers::{argsort_by, solve_lin_sys};
use crate::penalties::Penalty;

#[cfg(test)]
mod tests;

pub fn construct_grad<F, DF, DM, T>(
    dataset: &DatasetBase<DM, T>,
    Xw: ArrayView1<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
) -> Array1<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<F, DM, T>,
{
    let ws_size = ws.len();
    let mut grad = Array1::<F>::zeros(ws_size);
    for (idx, &j) in ws.iter().enumerate() {
        grad[idx] = datafit.gradient_j(dataset, Xw, j);
    }
    grad
}

pub fn kkt_violation<F, DF, P, DM, T>(
    dataset: &DatasetBase<DM, T>,
    w: ArrayView1<F>,
    Xw: ArrayView1<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
    penalty: &P,
) -> (Array1<F>, F)
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let grad_ws = construct_grad(dataset, Xw, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

pub fn construct_ws_from_kkt<F>(
    kkt: &mut Array1<F>,
    w: ArrayView1<F>,
    p0: usize,
) -> (Array1<usize>, usize)
where
    F: 'static + Float,
{
    let n_features = w.len();
    let mut nnz_features: usize = 0;

    for j in 0..n_features {
        if w[j] != F::zero() {
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

pub fn anderson_accel<F, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    w: &mut Array1<F>,
    Xw: &mut Array1<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
    last_K_w: &mut Array2<F>,
    U: &mut Array2<F>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let X = dataset.design_matrix;
    let y = dataset.targets;

    let n_samples = dataset.n_samples();

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

                let mut w_acc = Array1::<F>::zeros(w.len());

                // Extrapolation
                for (idx, &j) in ws.iter().enumerate() {
                    for k in 0..K {
                        w_acc[j] += last_K_w[[k, idx]] * c[k];
                    }
                }

                let mut Xw_acc = Array1::<F>::zeros(n_samples);

                match dataset.matrix_type() {
                    DesignMatrixType::Dense => {
                        for i in 0..Xw_acc.len() {
                            for &j in ws {
                                Xw_acc[i] += X[[i, j]] * w_acc[j];
                            }
                        }
                    }
                    DesignMatrixType::Sparse => {
                        for &j in ws {
                            for idx in X.indptr[j]..X.indptr[j + 1] {
                                Xw_acc[X.indices[idx as usize] as usize] +=
                                    X.data[idx as usize] * w_acc[j];
                            }
                        }
                    }
                }

                let p_obj = datafit.value(dataset, Xw.view()) + penalty.value(w.view());
                let p_obj_acc = datafit.value(dataset, Xw_acc.view()) + penalty.value(w_acc.view());

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

pub fn cd_epoch<F, D, DF, P, T>(
    dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    w: &mut Array1<F>,
    Xw: &mut Array1<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
) where
    F: 'static + Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<F, ArrayBase<D, Ix2>, T>,
    P: Penalty<F>,
{
    let n_samples = dataset.n_samples();
    let X = dataset.design_matrix;
    let lipschitz = datafit.lipschitz();

    for &j in ws {
        if lipschitz[j] == F::zero() {
            continue;
        }
        let old_w_j = w[j];
        let grad_j = datafit.gradient_j(dataset, Xw.view(), j);
        w[j] = penalty.prox_op(old_w_j - grad_j / lipschitz[j], F::one() / lipschitz[j]);
        if w[j] != old_w_j {
            for i in 0..n_samples {
                Xw[i] += (w[j] - old_w_j) * X[[i, j]];
            }
        }
    }
}

pub fn cd_epoch_sparse<'a, F, DF, P, T>(
    dataset: &DatasetBase<CSCArray<'a, F>, T>,
    w: &mut Array1<F>,
    Xw: &mut Array1<F>,
    datafit: &DF,
    penalty: &P,
    ws: ArrayView1<usize>,
) where
    F: 'static + Float,
    T: Targets<Elem = F>,
    DF: Datafit<F, CSCArray<'a, F>, T>,
    P: Penalty<F>,
{
    let lipschitz = datafit.lipschitz();
    let X = dataset.design_matrix;

    for &j in ws {
        if lipschitz[j] == F::zero() {
            continue;
        }
        let old_w_j = w[j];
        let grad_j = datafit.gradient_j(dataset, Xw.view(), j);
        w[j] = penalty.prox_op(old_w_j - grad_j / lipschitz[j], F::one() / lipschitz[j]);
        let diff = w[j] - old_w_j;
        if diff != F::zero() {
            for i in X.indptr[j]..X.indptr[j + 1] {
                Xw[X.indices[i as usize] as usize] += diff * X.data[i as usize];
            }
        }
    }
}

pub fn solver<F, DM, T, DF, P>(
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
) -> Array1<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();

    datafit.initialize(dataset);

    let X = dataset.design_matrix;
    let y = dataset.targets;

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if _p0 > n_features { n_features } else { _p0 };

    let mut w = Array1::<F>::zeros(n_features);
    let mut Xw = Array1::<F>::zeros(n_samples);

    for t in 0..max_iter {
        let (mut kkt, kkt_max) = kkt_violation(
            dataset,
            w.view(),
            Xw.view(),
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

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);

        let mut last_K_w = Array2::<F>::zeros((K + 1, ws_size));
        let mut U = Array2::<F>::zeros((K, ws_size));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            match dataset.matrix_type() {
                DesignMatrixType::Dense => {
                    cd_epoch(dataset, &mut w, &mut Xw, datafit, penalty, ws.view());
                }
                DesignMatrixType::Sparse => {
                    cd_epoch_sparse(dataset, &mut w, &mut Xw, datafit, penalty, ws.view());
                }
            }

            // Anderson acceleration
            if use_accel {
                anderson_accel(
                    dataset,
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
                let p_obj = datafit.value(dataset, Xw.view()) + penalty.value(w.view());

                let (_, kkt_ws_max) =
                    kkt_violation(dataset, w.view(), Xw.view(), ws.view(), datafit, penalty);

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

    w
}
