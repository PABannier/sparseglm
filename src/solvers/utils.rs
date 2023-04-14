/// This function constructs the gradient of a datafit restricted to the features
/// in the working set. It is used in [`opt_cond_violation`] to rank features included
/// in the working set.
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

/// This function computes the distance of the negative gradient of the datafit to the
/// subdifferential of the penalty restricted to the working set. It returns
/// an array containing the distances for each feature in the working set as well
/// as the maximum distance.
pub fn opt_cond_violation<F, DF, P, DM, T>(
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
/// in descending order of the features having the largest violation to the optimality
/// conditions. The inner coordinate descent solver then cycles through the working set
/// (a subset of the features in the design matrix).
pub fn construct_ws_from_kkt<F: 'static + Float>(
    kkt: &mut Array1<F>,
    w: ArrayView1<F>,
    ws_start_size: usize,
) -> (Array1<usize>, usize) {
    let n_features = w.len();
    let mut nnz_features: usize = 0;

    // Count features whose weights are non null and initializes their distance to be
    // infinity
    for j in 0..n_features {
        if w[j] != F::zero() {
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
