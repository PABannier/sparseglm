use ndarray::{Array1, Array2, ArrayView1};

use super::Float;
use crate::datafits::single_task::Datafit;
use crate::datasets::{AsSingleTargets, DatasetBase, DesignMatrix};
use crate::helpers::helpers::{argsort_by, solve_lin_sys};
use crate::penalties::separable::Penalty;

#[cfg(test)]
mod tests;

/// This function allows to construct the gradient of a datafit restricted to
/// the features present in the working set. It is used in [`kkt_violation`] to
/// rank features included in the working set.
pub fn construct_grad<F, DF, DM, T>(
    dataset: &DatasetBase<DM, T>,
    Xw: ArrayView1<F>,
    ws: ArrayView1<usize>,
    datafit: &DF,
) -> Array1<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
{
    Array1::from_iter(
        ws.iter()
            .map(|&j| datafit.gradient_j(dataset, Xw, j))
            .collect::<Vec<F>>(),
    )
}

/// This function computes the distance of the gradient of the datafit to the
/// subdifferential of the penalty restricted to the working set. It returns
/// an array containing the distances for each feature in the working set as well
/// as the maximum distance.
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
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let grad_ws = construct_grad(dataset, Xw, ws, datafit);
    let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
    (kkt_ws, kkt_ws_max)
}

/// This function is used to construct a working set by sorting the indices
/// of the features having the smallest distance between their gradient and
/// the subdifferential of the penalty. The inner coordinate descent solver then
/// cycles through the working set (a subset of the features in the design matrix).
pub fn construct_ws_from_kkt<F: 'static + Float>(
    kkt: &mut Array1<F>,
    w: ArrayView1<F>,
    p0: usize,
) -> (Array1<usize>, usize) {
    let n_features = w.len();
    let mut nnz_features: usize = 0;

    // Counts number of feature whose weights are non null and initializes their
    // distance to be infinity
    for j in 0..n_features {
        if w[j] != F::zero() {
            nnz_features += 1;
            kkt[j] = F::infinity();
        }
    }

    // Geometric growth of the working set size
    let ws_size = usize::max(p0, usize::min(2 * nnz_features, n_features));

    // Sort indices by descending order (argmin)
    let mut sorted_indices = argsort_by(&kkt, |a, b| {
        // Swapped order for sorting in descending order
        b.partial_cmp(a).expect("Elements must not be NaN.")
    });
    sorted_indices.truncate(ws_size);

    let ws = Array1::from_shape_vec(ws_size, sorted_indices).unwrap();
    (ws, ws_size)
}

/// This function performs Anderson acceleration given K previous iterates of
/// coordinate descent cycles. Anderson acceleration is a non-linear extrapolation
/// technique that finds extrapolated points during the descent.
///
/// The Anderson extrapolation schema is guaranteed to converge, since the extrapolated
/// point is selected if and only if the value of the objective at this extrapolated
/// point has decreased compared to the previous non-extrapolated point.
///
/// Reference: `https://arxiv.org/pdf/2011.10065.pdf`
pub fn anderson_accel<F, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    datafit: &DF,
    penalty: &P,
    last_K_w: &mut Array2<F>,
    U: &mut Array2<F>,
    w: &mut Array1<F>,
    Xw: &mut Array1<F>,
    ws: ArrayView1<usize>,
    epoch: usize,
    K: usize,
    verbose: bool,
) where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();

    // Update the last_K_w array to hold the up-to-date weight vectors
    ws.iter().enumerate().for_each(|(idx, &j)| {
        last_K_w[[epoch % (K + 1), idx]] = w[j];
    });

    if epoch % (K + 1) == K {
        // for k in 0..K {
        //     for j in 0..ws.len() {
        //         U[[k, j]] = last_K_w[[k + 1, j]] - last_K_w[[k, j]];
        //     }
        // }
        *U = Array2::from_shape_vec(
            (K, ws.len()),
            last_K_w
                .rows()
                .into_iter()
                .take(K)
                .zip(last_K_w.rows().into_iter().skip(1))
                .map(|(row_k, row_k_plus_1)| &row_k_plus_1 - &row_k)
                .collect::<Vec<Array1<F>>>()
                .into_iter()
                .flatten()
                .collect(),
        )
        .unwrap();

        // Computes the extrapolation matrix à la Anderson
        let C = U.t().dot(U);
        let _res = solve_lin_sys(C.view(), Array1::<F>::ones(K).view());

        match _res {
            Ok(z) => {
                let c = &z / z.sum();

                // Computes the extrapolated point
                let mut w_acc = Array1::<F>::zeros(n_features);
                last_K_w
                    .rows()
                    .into_iter()
                    .take(K)
                    .zip(c)
                    .map(|(row, c_k)| &row * c_k)
                    .fold(Array1::<F>::zeros(ws.len()), std::ops::Add::add)
                    .iter()
                    .zip(ws)
                    .for_each(|(&extrapolated_pt_j, &j)| {
                        w_acc[j] = extrapolated_pt_j;
                    });

                // Computes the extrapolated datafit
                let Xw_acc = dataset
                    .design_matrix()
                    .compute_extrapolated_fit(ws, &w_acc, n_samples);

                // Computes the objective value at the extrapolated point and at the
                // non-extrapolated point
                let p_obj = datafit.value(dataset, Xw.view()) + penalty.value(w.view());
                let p_obj_acc = datafit.value(dataset, Xw_acc.view()) + penalty.value(w_acc.view());

                // Compares the objective value at the extrapolated point and the objective
                // value at the non-extrapolated point. The extrapolated point is retained
                // if and only if it decreases the objective. This ensures that the routine
                // converges.
                if p_obj_acc < p_obj {
                    w.assign(&w_acc);
                    Xw.assign(&Xw_acc);

                    if verbose {
                        println!("[ACCEL] p_obj {:#?} :: p_obj_acc {:#?}", p_obj, p_obj_acc);
                    }
                }
            }
            // Some extrapolation matrix can be very ill-conditioned which makes
            // the inversion computationally intractable (inf or NaN) values.
            // This is expected, thus the non-panicking way of handling the error.
            Err(_) => {
                if verbose {
                    println!("---- Singular extrapolation matrix. Could not extrapolate.");
                }
            }
        }
    }
}

/// This is the backbone function for the [`sparseglm`] crate. It implements
/// the usual coordinate descent optimization routine using working sets. This
/// routine is composed is composed of two nested loops.
///
/// The outer loop is used to progressively increase the size of the working
/// set until a specific suboptimality threshold is reached. This outer loop
/// makes the working set size grow in a geometric fashion and selects the
/// features in the design matrix whose gradient is the closest from the
/// subdifferential of the penalty by calling [`kkt_violation`] and
/// [`construct_ws_from_kkt`] functions. This loop runs for a fixed number of
/// iterations.
///
/// The inner loop is to solve the optimization problem restricted to the working set.
/// Inside this loop, regular calls are made to [`anderson_accel`] to try to find an
/// extrapolated point that would speed up the convergence of the algorithm.
/// It then calls [`cd_epoch`], a method in a [`Solver`] struct that carries out
/// the actual coordinate descent, by cycling through the features in the
/// working set.
///
/// Note that combining working set and Anderson acceleration is a powerful
/// technique as once the support has been identified (for a given level of
/// regularization, the support is the set of features whose weights are non-
/// null), Anderson acceleration kicks in and allows to find extrapolated points
/// thus dramatically speeding the convergence speed.
pub fn coordinate_descent<F, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    datafit: &mut DF,
    penalty: &P,
    p0: usize,
    max_iterations: usize,
    max_epochs: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
) -> Array1<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();

    // Pre-computes the Lipschitz constants and the matrix-vector Xty product
    // that is later used in the optimization procedure.
    datafit.initialize(dataset);

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    // The starting working set can't be greater than the number of features
    let p0 = if p0 > n_features { n_features } else { p0 };

    let mut w = Array1::<F>::zeros(n_features);
    let mut Xw = Array1::<F>::zeros(n_samples);

    // Outer loop in charge of constructing the working set
    for t in 0..max_iterations {
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
        if kkt_max <= tolerance {
            break;
        }

        // Construct the working set based on previously computed KKT violation
        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);

        let mut last_K_w = Array2::<F>::zeros((K + 1, ws_size));
        let mut U = Array2::<F>::zeros((K, ws_size));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        // Inner loop that implements the actual coordinate descent routine
        for epoch in 0..max_epochs {
            let lipschitz = datafit.step_size();

            // Cycle through the features in the working set
            for &j in ws.iter() {
                match lipschitz[j] == F::zero() {
                    true => continue,
                    false => {
                        let old_w_j = w[j];
                        let grad_j = datafit.gradient_j(dataset, Xw.view(), j);
                        w[j] =
                            penalty.prox(old_w_j - grad_j / lipschitz[j], F::one() / lipschitz[j]);

                        let diff = w[j] - old_w_j;
                        if diff != F::zero() {
                            dataset.design_matrix().update_model_fit(&mut Xw, diff, j);
                        }
                    }
                }
            }

            // Attempt to find an extrapolated point using Anderson acceleration
            if use_acceleration {
                anderson_accel(
                    dataset,
                    datafit,
                    penalty,
                    &mut last_K_w,
                    &mut U,
                    &mut w,
                    &mut Xw,
                    ws.view(),
                    epoch,
                    K,
                    verbose,
                );
            }

            // Check that the maximum distance between the gradient of the datafit
            // and the subdifferential of the penalty is smaller than the tolerance
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

                match ws_size == n_features {
                    // If it is, we stop the optimization procedure
                    true => {
                        if kkt_ws_max <= tolerance {
                            break;
                        }
                    }
                    false => {
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
    }

    w
}
