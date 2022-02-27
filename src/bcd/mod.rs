extern crate ndarray;

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::Float;
use crate::datafits_multitask::MultiTaskDatafit;
use crate::datasets::DesignMatrix;
use crate::datasets::{AsMultiTargets, DatasetBase};
use crate::estimators::hyperparams::SolverParams;
use crate::helpers::helpers::{argsort_by, solve_lin_sys};
use crate::penalties_multitask::PenaltyMultiTask;
use crate::solver_multitask::{BCDSolver, MultiTaskExtrapolator};

#[cfg(test)]
mod tests;

pub fn construct_grad<F, DF, DM, T>(
    dataset: &DatasetBase<DM, T>,
    XW: ArrayView2<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
) -> Array2<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
{
    let ws_size = ws.len();
    let n_tasks = dataset.targets().n_tasks();
    let mut grad = Array2::<F>::zeros((ws_size, n_tasks));
    for (idx, &j) in ws.iter().enumerate() {
        let grad_j = datafit.gradient_j(&dataset, XW, j);
        grad.slice_mut(s![idx, ..]).assign(&grad_j);
    }
    grad
}

pub fn kkt_violation<F, DF, P, DM, T>(
    dataset: &DatasetBase<DM, T>,
    W: ArrayView2<F>,
    XW: ArrayView2<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
    penalty: &P,
) -> (Array1<F>, F)
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: PenaltyMultiTask<F>,
{
    let grad_ws = construct_grad(dataset, XW, ws, datafit);
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
        if W.slice(s![j, ..]).fold(F::zero(), |sum, &x| sum + x.abs()) != F::zero() {
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

pub fn anderson_accel<F, DM, T, DF, P, S>(
    dataset: &DatasetBase<DM, T>,
    solver: &S,
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
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: PenaltyMultiTask<F>,
    S: BCDSolver<F, DF, P, DM, T> + MultiTaskExtrapolator<F, DM, T>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();
    let n_tasks = dataset.targets().n_tasks();

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

        let C = U.t().dot(U);
        let _res = solve_lin_sys(C.view(), Array1::<F>::ones(K).view()); // TODO: change into LAPACK inversion

        match _res {
            Ok(z) => {
                let c = &z / z.sum();

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
                solver.extrapolate(dataset, &mut XW_acc, W_acc.view(), ws);

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
                    println!("---- Warning: Singular extrapolation matrix.");
                }
            }
        }
    }
}

pub fn block_coordinate_descent<F, DM, T, DF, P, S>(
    dataset: &DatasetBase<DM, T>,
    datafit: &mut DF,
    solver: &S,
    penalty: &P,
    params: SolverParams<F>,
) -> Array2<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: PenaltyMultiTask<F>,
    S: BCDSolver<F, DF, P, DM, T> + MultiTaskExtrapolator<F, DM, T>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();
    let n_tasks = dataset.targets().n_tasks();

    datafit.initialize(dataset);

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if params.p0 > n_features {
        n_features
    } else {
        params.p0
    };

    let mut W = Array2::<F>::zeros((n_features, n_tasks));
    let mut XW = Array2::<F>::zeros((n_samples, n_tasks));

    for t in 0..params.max_iter {
        let (mut kkt, kkt_max) = kkt_violation(
            dataset,
            W.view(),
            XW.view(),
            all_feats.view(),
            datafit,
            penalty,
        );

        if params.verbose {
            println!("KKT max violation: {:#?}", kkt_max);
        }
        if kkt_max <= params.tol {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, W.view(), p0);

        let mut last_K_W = Array2::<F>::zeros((params.K + 1, ws_size * n_tasks));
        let mut U = Array2::<F>::zeros((params.K, ws_size * n_tasks));

        if params.verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..params.max_epochs {
            solver.bcd_epoch(dataset, datafit, penalty, &mut W, &mut XW, ws.view());

            // Anderson acceleration
            if params.use_accel {
                anderson_accel(
                    dataset,
                    solver,
                    &mut W,
                    &mut XW,
                    datafit,
                    penalty,
                    ws.view(),
                    &mut last_K_W,
                    &mut U,
                    epoch,
                    params.K,
                    params.verbose,
                );
            }

            // KKT violation check
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());

                let (_, kkt_ws_max) =
                    kkt_violation(dataset, W.view(), XW.view(), ws.view(), datafit, penalty);

                if params.verbose {
                    println!(
                        "epoch: {} :: obj: {:#?} :: kkt: {:#?}",
                        epoch, p_obj, kkt_ws_max
                    );
                }

                if ws_size == n_features {
                    if kkt_ws_max <= params.tol {
                        break;
                    }
                } else {
                    if kkt_ws_max < F::cast(0.3) * kkt_max {
                        if params.verbose {
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
