use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};

use super::Float;
use crate::datafits::multi_task::MultiTaskDatafit;
use crate::datasets::DesignMatrix;
use crate::datasets::{AsMultiTargets, DatasetBase};
use crate::penalties::block_separable::MultiTaskPenalty;
use crate::utils::helpers::{argsort_by, solve_lin_sys};

#[cfg(test)]
mod tests;

/// This function allows to construct the gradient of a datafit restricted to
/// the features present in the working set. It is used in [`opt_cond_violation`] to
/// rank features included in the working set.
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

/// This function computes the distance of the gradient of the datafit to the
/// subdifferential of the penalty restricted to the working set. It returns
/// an array containing the distances for each feature in the working set as well
/// as the maximum distance.
pub fn opt_cond_violation<F, DF, P, DM, T>(
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
    P: MultiTaskPenalty<F>,
{
    let grad_ws = construct_grad(dataset, XW, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(W, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

/// This function is used to construct a working set by sorting the indices
/// of the features having the smallest distance between their gradient and
/// the subdifferential of the penalty. The inner block coordinate descent solver then
/// cycles through the working set (a subset of the features in the design matrix).
pub fn construct_ws_from_kkt<F>(
    kkt: &mut Array1<F>,
    W: ArrayView2<F>,
    ws_start_size: usize,
) -> (Array1<usize>, usize)
where
    F: 'static + Float,
{
    let n_features = W.shape()[0];
    let mut nnz_features: usize = 0;

    // Counts number of feature whose weights have a non-null norm and initializes
    // the distance to be infinity
    for j in 0..n_features {
        if W.slice(s![j, ..]).iter().any(|&x| x != F::zero()) {
            nnz_features += 1;
            kkt[j] = F::infinity();
        }
    }

    // Geometric growth of the working set size
    let ws_size = usize::max(ws_start_size, usize::min(2 * nnz_features, n_features));

    // Sort indices by descending order (argmin)
    let mut sorted_indices = argsort_by(&kkt, |a, b| {
        // Swapped order for sorting in descending order
        b.partial_cmp(a).expect("Elements must not be NaN.")
    });
    sorted_indices.truncate(ws_size);

    let ws = Array1::from_shape_vec(ws_size, sorted_indices).unwrap();
    (ws, ws_size)
}

/// This function is a multi-task variant of the [`anderson_accel`] function in
/// the single-task case.
pub fn anderson_accel<F, DM, T, DF, P>(
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
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: MultiTaskPenalty<F>,
{
    let n_samples = dataset.targets().n_samples();
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

        // Computes the extrapolation matrix à la Anderson
        let C = U.t().dot(U);
        let _res = solve_lin_sys(C.view(), Array1::<F>::ones(K).view()); // TODO: change into LAPACK inversion

        match _res {
            Ok(z) => {
                let c = &z / z.sum();

                // Computes the extrapolated point
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

                // Computes the objective value at the extrapolated point and at the
                // non-extrapolated point
                let XW_acc = dataset
                    .design_matrix()
                    .compute_extrapolated_fit_multi_task(ws, &W_acc, n_samples, n_tasks);
                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());
                let p_obj_acc = datafit.value(dataset, XW_acc.view()) + penalty.value(W_acc.view());

                // Compares the objective value at the extrapolated point and the objective
                // value at the non-extrapolated point. The extrapolated point is retained
                // if and only if it decreases the objective. This ensures that the routine
                // converges.
                if p_obj_acc < p_obj {
                    W.assign(&W_acc);
                    XW.assign(&XW_acc);

                    if verbose {
                        println!("[ACCEL] obj {:#?} :: obj_acc {:#?}", p_obj, p_obj_acc);
                    }
                }
            }
            // Some extrapolation matrix can be very ill-conditioned which makes
            // the inversion computationally intractable (inf or NaN) values.
            // This is expected, thus the non-panicking way of handling the error.
            Err(_) => {
                if verbose {
                    println!("---- Warning: Singular extrapolation matrix.");
                }
            }
        }
    }
}

/// This is the backbone function for the crate, in the multi-task case. For a
/// detailed description, see [`coordinate_descent`] function.
pub fn block_coordinate_descent<F, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    datafit: &mut DF,
    penalty: &P,
    ws_start_size: usize,
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
    P: MultiTaskPenalty<F>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();
    let n_tasks = dataset.targets().n_tasks();

    // Pre-computes the Lipschitz constants and the matrix-matrix XTY product
    // that is later used in the optimization procedure.
    datafit.initialize(dataset);

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    // The starting working set can't be greater than the number of features
    let ws_start_size = if ws_start_size > n_features {
        n_features
    } else {
        ws_start_size
    };

    let mut W = Array2::<F>::zeros((n_features, n_tasks));
    let mut XW = Array2::<F>::zeros((n_samples, n_tasks));

    // Outer loop in charge of constructing the working set
    for t in 0..max_iterations {
        let (mut kkt, kkt_max) = opt_cond_violation(
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

        // Construct the working set based on previously computed KKT violation
        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, W.view(), ws_start_size);

        let mut last_K_W = Array2::<F>::zeros((K + 1, ws_size * n_tasks));
        let mut U = Array2::<F>::zeros((K, ws_size * n_tasks));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        // Inner loop that implements the actual block coordinate descent routine
        for epoch in 0..max_epochs {
            let lipschitz = datafit.step_size();

            // Cycle through the features in the working set
            for &j in ws.iter() {
                match lipschitz[j] == F::zero() {
                    true => continue,
                    false => {
                        let old_W_j = W.slice(s![j, ..]).to_owned(); // Idea: pre-slice W and store columns
                        let grad_j = datafit.gradient_j(dataset, XW.view(), j);

                        let step = &old_W_j - grad_j / lipschitz[j];
                        let upd = penalty.prox(step.view(), F::one() / lipschitz[j]);

                        // W.slice_mut(s![j, ..]).assign(&upd);
                        // For loops are way faster than chaining slice_mut and assign
                        for t in 0..n_tasks {
                            W[[j, t]] = upd[t];
                        }

                        let diff = Array1::from_iter(
                            old_W_j
                                .iter()
                                .enumerate()
                                .map(|(t, &old_w_jt)| W[[j, t]] - old_w_jt)
                                .collect::<Vec<F>>(),
                        );

                        if diff.iter().any(|&x| x != F::zero()) {
                            dataset.design_matrix().update_model_fit_multi_task(
                                &mut XW,
                                diff.view(),
                                j,
                            );
                        }
                    }
                }
            }

            // Attempt to find an extrapolated point using Anderson acceleration
            if use_acceleration {
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

            // Check that the maximum distance between the gradient of the datafit
            // and the subdifferential of the penalty is smaller than the tolerance
            if epoch > 0 && epoch % 10 == 0 {
                let p_obj = datafit.value(dataset, XW.view()) + penalty.value(W.view());

                let (_, kkt_ws_max) =
                    opt_cond_violation(dataset, W.view(), XW.view(), ws.view(), datafit, penalty);

                if verbose {
                    println!(
                        "epoch: {} :: obj: {:#?} :: kkt: {:#?}",
                        epoch, p_obj, kkt_ws_max
                    );
                }

                if ws_size == n_features {
                    // If it is, we stop the optimization procedure
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
