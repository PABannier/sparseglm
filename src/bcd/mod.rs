extern crate ndarray;

use ndarray::{Array1, Array2, ArrayView1, Ix2};

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::DesignMatrix;
use crate::datasets::{DatasetBase, Targets};
use crate::helpers::helpers::solve_lin_sys;
use crate::penalties::Penalty;
use crate::solvers::{BCDSolver, Extrapolator, WorkingSet};

#[cfg(test)]
mod tests;

pub fn anderson_accel<'a, F, DM, T, DF, P, S>(
    dataset: &'a DatasetBase<DM, T>,
    solver: &'a S,
    W: &'a mut Array2<F>,
    XW: &'a mut Array2<F>,
    datafit: &'a DF,
    penalty: &'a P,
    ws: ArrayView1<'a, usize>,
    last_K_W: &'a mut Array2<F>,
    U: &'a mut Array2<F>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<'a, F, DM, T, Ix2>,
    P: Penalty<'a, F, Ix2>,
    S: BCDSolver<'a, F, DF, P, DM, T> + Extrapolator<'a, F, DM, T, Ix2>,
{
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();
    let n_tasks = dataset.n_tasks();

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
                    println!("Singular extrapolation matrix. Skipped.");
                }
            }
        }
    }
}

pub fn block_coordinate_descent<'a, F, DM, T, DF, P, S>(
    dataset: &'a DatasetBase<DM, T>,
    datafit: &'a mut DF,
    solver: &'a S,
    penalty: &'a P,
    max_iter: usize,
    max_epochs: usize,
    _p0: usize,
    tol: F,
    use_accel: bool,
    K: usize,
    verbose: bool,
) -> Array2<F>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    DF: Datafit<'a, F, DM, T, Ix2>,
    P: Penalty<'a, F, Ix2>,
    S: BCDSolver<'a, F, DF, P, DM, T>
        + Extrapolator<'a, F, DM, T, Ix2>
        + WorkingSet<'a, F, DF, P, DM, T, Ix2>,
{
    let n_samples = dataset.n_samples();
    let n_features = dataset.n_features();
    let n_tasks = dataset.n_tasks();

    datafit.initialize(dataset);

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if _p0 > n_features { n_features } else { _p0 };

    let mut W = Array2::<F>::zeros((n_features, n_tasks));
    let mut XW = Array2::<F>::zeros((n_samples, n_tasks));

    for t in 0..max_iter {
        let (mut kkt, kkt_max) = solver.kkt_violation(
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

        let (ws, ws_size) = solver.construct_ws_from_kkt(&mut kkt, W.view(), p0);

        let mut last_K_W = Array2::<F>::zeros((K + 1, ws_size * n_tasks));
        let mut U = Array2::<F>::zeros((K, ws_size * n_tasks));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            solver.bcd_epoch(dataset, datafit, penalty, &mut W, &mut XW, ws.view());

            // Anderson acceleration
            if use_accel {
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
                    K,
                    verbose,
                );
            }

            // KKT violation check
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());

                let (_, kkt_ws_max) =
                    solver.kkt_violation(dataset, W.view(), XW.view(), ws.view(), datafit, penalty);

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
