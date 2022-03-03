use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::Float;
use crate::datafits_multitask::MultiTaskDatafit;
use crate::datasets::DesignMatrix;
use crate::datasets::{AsMultiTargets, DatasetBase};
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
    Array2::from_shape_vec(
        (ws.len(), dataset.targets().n_tasks()),
        ws.iter()
            .map(|&j| datafit.gradient_j(&dataset, XW, j))
            .collect::<Vec<Array1<F>>>()
            .into_iter()
            .flatten()
            .collect(),
    )
    .unwrap()
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
        if W.slice(s![j, ..]).iter().any(|&x| x != F::zero()) {
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
    let n_features = dataset.design_matrix().n_features();
    let n_tasks = dataset.targets().n_tasks();

    // last_K_w[epoch % (K + 1)] = W[ws, :].ravel()
    last_K_W
        .slice_mut(s![epoch % (K + 1), ..])
        .assign(&Array1::from_iter(
            ws.iter()
                .map(|&j| W.slice(s![j, ..]).to_owned()) // TODO: Circumvent this to_owned
                .collect::<Vec<Array1<F>>>()
                .into_iter()
                .flatten()
                .collect::<Vec<F>>(),
        ));

    if epoch % (K + 1) == K {
        // for k in 0..K {
        //     for j in 0..(ws.len() * n_tasks) {
        //         U[[k, j]] = last_K_W[[k + 1, j]] - last_K_W[[k, j]];
        //     }
        // }
        *U = Array2::from_shape_vec(
            (K, ws.len() * n_tasks),
            last_K_W
                .rows()
                .into_iter()
                .take(K)
                .zip(last_K_W.rows().into_iter().skip(1))
                .map(|(row_k, row_k_plus_1)| &row_k_plus_1 - &row_k)
                .collect::<Vec<Array1<F>>>()
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();

        let C = U.t().dot(U);
        let _res = solve_lin_sys(C.view(), Array1::<F>::ones(K).view()); // TODO: change into LAPACK inversion

        match _res {
            Ok(z) => {
                let c = &z / z.sum();

                // Extrapolation
                let mut W_acc = Array2::<F>::zeros((n_features, n_tasks));
                last_K_W
                    .rows()
                    .into_iter()
                    .take(K)
                    .zip(c)
                    .map(|(row, c_k)| &row * c_k)
                    .fold(Array1::<F>::zeros(ws.len() * n_tasks), std::ops::Add::add)
                    .into_shape((ws.len(), n_tasks))
                    .unwrap()
                    .rows()
                    .into_iter()
                    .zip(ws)
                    .for_each(|(row, &j)| {
                        W_acc.slice_mut(s![j, ..]).assign(&row);
                    });

                let XW_acc = solver.extrapolate(dataset, W_acc.view(), ws);
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
    p0: usize,
    max_iterations: usize,
    max_epochs: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
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

    let p0 = if p0 > n_features { n_features } else { p0 };

    let mut W = Array2::<F>::zeros((n_features, n_tasks));
    let mut XW = Array2::<F>::zeros((n_samples, n_tasks));

    for t in 0..max_iterations {
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
        if kkt_max <= tolerance {
            break;
        }

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, W.view(), p0);

        let mut last_K_W = Array2::<F>::zeros((K + 1, ws_size * n_tasks));
        let mut U = Array2::<F>::zeros((K, ws_size * n_tasks));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            solver.bcd_epoch(dataset, datafit, penalty, &mut W, &mut XW, ws.view());

            // Anderson acceleration
            if use_acceleration {
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
                    kkt_violation(dataset, W.view(), XW.view(), ws.view(), datafit, penalty);

                if verbose {
                    println!(
                        "epoch: {} :: obj: {:#?} :: kkt: {:#?}",
                        epoch, p_obj, kkt_ws_max
                    );
                }

                if ws_size == n_features {
                    if kkt_ws_max <= tolerance {
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
