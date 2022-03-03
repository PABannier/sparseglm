use ndarray::{Array1, Array2, ArrayView1};

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::{AsSingleTargets, DatasetBase, DesignMatrix};
use crate::helpers::helpers::argsort_by;
use crate::penalties::Penalty;
use crate::solver::{CDSolver, Extrapolator};

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
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
{
    Array1::from_iter(
        ws.iter()
            .map(|&j| datafit.gradient_j(dataset, Xw, j))
            .collect::<Vec<F>>(),
    )
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
    T: AsSingleTargets<Elem = F>,
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

pub fn anderson_accel<F, DM, T, DF, P, S>(
    dataset: &DatasetBase<DM, T>,
    datafit: &DF,
    penalty: &P,
    solver: &S,
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
    S: Extrapolator<F, DM, T>,
{
    let n_features = dataset.design_matrix().n_features();

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

        let C = U.t().dot(U);
        let _res = C.invc();

        match _res {
            Ok(C_inv) => {
                let z = Array1::from_iter(
                    C_inv
                        .rows()
                        .into_iter()
                        .map(|row| row.sum())
                        .collect::<Vec<F>>(),
                );
                let c = &z / z.sum();

                // Extrapolation
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

                let Xw_acc = solver.extrapolate(dataset, w_acc.view(), ws);
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
                    println!("---- Warning: Singular extrapolation matrix.");
                }
            }
        }
    }
}

pub fn coordinate_descent<F, DM, T, DF, P, S>(
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
) -> Array1<F>
where
    F: 'static + Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
    S: CDSolver<F, DF, P, DM, T> + Extrapolator<F, DM, T>,
{
    let n_samples = dataset.targets().n_samples();
    let n_features = dataset.design_matrix().n_features();

    datafit.initialize(dataset);

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    let p0 = if p0 > n_features { n_features } else { p0 };

    let mut w = Array1::<F>::zeros(n_features);
    let mut Xw = Array1::<F>::zeros(n_samples);

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

        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), p0);

        let mut last_K_w = Array2::<F>::zeros((K + 1, ws_size));
        let mut U = Array2::<F>::zeros((K, ws_size));

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        for epoch in 0..max_epochs {
            solver.cd_epoch(dataset, datafit, penalty, &mut w, &mut Xw, ws.view());

            // Anderson acceleration
            if use_acceleration {
                anderson_accel(
                    dataset,
                    datafit,
                    penalty,
                    solver,
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

                match ws_size == n_features {
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
