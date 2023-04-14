use ndarray::{Array1, Array2, ArrayView1};

use super::Float;
use crate::datafits::single_task::Datafit;
use crate::datasets::{AsSingleTargets, DatasetBase, DesignMatrix};
use crate::penalties::separable::Penalty;
use utils::{construct_ws_from_kkt, opt_cond_violation};

#[cfg(test)]
mod tests;

const EPS_TOL: f32 = 0.3;

pub fn prox_newton<F, DM, T, DF, P>(
    dataset: &DatasetBase<DM, T>,
    datafit: &mut DF,
    penalty: &P,
    ws_start_size: usize,
    max_iterations: usize,
    max_pn_iterations: usize,
    tolerance: F,
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
    let n_features = dataset.design_matrix.n_features();

    let all_feats = Array1::from_shape_vec(n_features, (0..n_features).collect()).unwrap();

    // The starting working set can't be greater than the number of features
    let ws_start_size = if ws_start_size > n_features {
        n_features
    } else {
        ws_start_size
    };

    let mut w = Array1::<F>::zeros(n_features);
    let mut Xw = Array1::<F>::zeros(n_samples);

    let stopping_criterion = 0.;

    // Outer loop in charge of constructing the working set
    for t in 0..max_iterations {
        let (mut kkt, kkt_max) = opt_cond_violation(
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
        let (ws, ws_size) = construct_ws_from_kkt(&mut kkt, w.view(), ws_start_size);

        if verbose {
            println!("Iteration {}, {} features in subproblem.", t + 1, ws_size);
        }

        // Inner Prox-Newton loop
        for pn_iter in 0..max_pn_iterations {}
    }
}
