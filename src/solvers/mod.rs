extern crate ndarray;

use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, Data, Dimension, Ix1, Ix2, OwnedRepr, ViewRepr,
};

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};
use crate::helpers::helpers::argsort_by;
use crate::penalties::Penalty;

#[cfg(test)]
mod tests;

pub struct Solver {}

impl Solver {
    pub fn new() -> Solver {
        Solver {}
    }
}

pub trait CDSolver<'a, F, DF, P, DM, T>
where
    F: Float,
    DF: Datafit<'a, F, DM, T, Ix1>,
    P: Penalty<'a, F, Ix1>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        datafit: &'a DF,
        penalty: &'a P,
        w: &'a mut Array1<F>,
        Xw: &'a mut Array1<F>,
        ws: ArrayView1<'a, usize>,
    );
}

pub trait BCDSolver<'a, F, DF, P, DM, T>
where
    F: Float,
    DF: Datafit<'a, F, DM, T, Ix2>,
    P: Penalty<'a, F, Ix2>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        datafit: &'a DF,
        penalty: &'a P,
        W: &'a mut Array2<F>,
        XW: &'a mut Array2<F>,
        ws: ArrayView1<'a, usize>,
    );
}

pub trait Extrapolator<'a, F, DM, T, I>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    I: Dimension,
{
    fn extrapolate(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        Xw_acc: &'a mut ArrayBase<OwnedRepr<F>, I>,
        w_acc: ArrayBase<ViewRepr<&'a F>, I>,
        ws: ArrayView1<'a, usize>,
    );
}

pub trait WorkingSet<'a, F, DF, P, DM, T, I>
where
    F: Float,
    DF: Datafit<'a, F, DM, T, I>,
    P: Penalty<'a, F, I>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    I: Dimension,
{
    fn kkt_violation(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        w: ArrayBase<ViewRepr<&'a F>, I>,
        Xw: ArrayBase<ViewRepr<&'a F>, I>,
        ws: ArrayView1<'a, usize>,
        datafit: &'a DF,
        penalty: &'a P,
    ) -> (Array1<F>, F);

    fn construct_ws_from_kkt(
        &self,
        kkt: &'a mut Array1<F>,
        w: ArrayBase<ViewRepr<&'a F>, I>,
        p0: usize,
    ) -> (Array1<usize>, usize);
}

/// This implementation block implements the coordinate descent epoch for dense
/// design matrices.

impl<'a, F, D, DF, P, T> CDSolver<'a, F, DF, P, ArrayBase<D, Ix2>, T> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    DF: Datafit<'a, F, ArrayBase<D, Ix2>, T, Ix1, Output = F>,
    P: Penalty<'a, F, Ix1, Input = F, Output = F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        datafit: &'a DF,
        penalty: &'a P,
        w: &'a mut Array1<F>,
        Xw: &'a mut Array1<F>,
        ws: ArrayView1<'a, usize>,
    ) {
        let n_samples = dataset.n_samples();
        let X = dataset.design_matrix();
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
}

/// This implementation block implements the coordinate descent epoch for sparse
/// design matrices.

impl<'a, F, DF, P, T> CDSolver<'a, F, DF, P, CSCArray<'a, F>, T> for Solver
where
    F: Float,
    DF: Datafit<'a, F, CSCArray<'a, F>, T, Ix1, Output = F>,
    P: Penalty<'a, F, Ix1, Input = F, Output = F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &'a DatasetBase<CSCArray<'a, F>, T>,
        datafit: &'a DF,
        penalty: &'a P,
        w: &'a mut Array1<F>,
        Xw: &'a mut Array1<F>,
        ws: ArrayView1<'a, usize>,
    ) {
        let lipschitz = datafit.lipschitz();
        let X = dataset.design_matrix();

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
}

/// This implementation block implements the extrapolation method for coordinate
/// descent solvers, using dense matrices.
///
impl<'a, F, D, T> Extrapolator<'a, F, ArrayBase<D, Ix2>, T, Ix1> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw_acc: &'a mut ArrayBase<OwnedRepr<F>, Ix1>,
        w_acc: ArrayBase<ViewRepr<&'a F>, Ix1>,
        ws: ArrayView1<'a, usize>,
    ) {
        let X = dataset.design_matrix();
        for i in 0..Xw_acc.len() {
            for &j in ws {
                Xw_acc[i] += X[[i, j]] * w_acc[j];
            }
        }
    }
}

/// This implementation block implements the extrapolation method for coordinate
/// descent solvers, using sparse matrices.
///
impl<'a, F, T> Extrapolator<'a, F, CSCArray<'a, F>, T, Ix1> for Solver
where
    F: Float,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &'a DatasetBase<CSCArray<'a, F>, T>,
        Xw_acc: &'a mut ArrayBase<OwnedRepr<F>, Ix1>,
        w_acc: ArrayBase<ViewRepr<&'a F>, Ix1>,
        ws: ArrayView1<'a, usize>,
    ) {
        let X = dataset.design_matrix();
        for &j in ws {
            for idx in X.indptr[j]..X.indptr[j + 1] {
                Xw_acc[X.indices[idx as usize] as usize] += X.data[idx as usize] * w_acc[j];
            }
        }
    }
}

/// This implementation block implements the coordinate descent epoch for dense
/// design matrices.

impl<'a, F, D, DF, P, T> BCDSolver<'a, F, DF, P, ArrayBase<D, Ix2>, T> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    DF: Datafit<'a, F, ArrayBase<D, Ix2>, T, Ix2, Output = ArrayBase<OwnedRepr<F>, Ix1>>,
    P: Penalty<
        'a,
        F,
        Ix2,
        Input = ArrayBase<ViewRepr<&'a F>, Ix1>,
        Output = ArrayBase<OwnedRepr<F>, Ix1>,
    >,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        datafit: &'a DF,
        penalty: &'a P,
        W: &'a mut Array2<F>,
        XW: &'a mut Array2<F>,
        ws: ArrayView1<'a, usize>,
    ) {
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();
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
}

/// This implementation block implements the coordinate descent epoch for sparse
/// design matrices.

impl<'a, F, DF, P, T> BCDSolver<'a, F, DF, P, CSCArray<'a, F>, T> for Solver
where
    F: Float,
    DF: Datafit<'a, F, CSCArray<'a, F>, T, Ix2, Output = ArrayBase<OwnedRepr<F>, Ix1>>,
    P: Penalty<
        'a,
        F,
        Ix2,
        Input = ArrayBase<ViewRepr<&'a F>, Ix1>,
        Output = ArrayBase<OwnedRepr<F>, Ix1>,
    >,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &'a DatasetBase<CSCArray<'a, F>, T>,
        datafit: &'a DF,
        penalty: &'a P,
        W: &'a mut Array2<F>,
        XW: &'a mut Array2<F>,
        ws: ArrayView1<'a, usize>,
    ) {
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();
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
}

/// This implementation block implements the extrapolation method for coordinate
/// descent solvers, using dense matrices.
///
impl<'a, F, D, T> Extrapolator<'a, F, ArrayBase<D, Ix2>, T, Ix2> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, T>,
        XW_acc: &'a mut ArrayBase<OwnedRepr<F>, Ix2>,
        W_acc: ArrayBase<ViewRepr<&'a F>, Ix2>,
        ws: ArrayView1<'a, usize>,
    ) {
        let X = dataset.design_matrix();
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();
        for i in 0..n_samples {
            for &j in ws {
                for t in 0..n_tasks {
                    XW_acc[[i, t]] += X[[i, j]] * W_acc[[j, t]];
                }
            }
        }
    }
}

/// This implementation block implements the extrapolation method for coordinate
/// descent solvers, using sparse matrices.
///
impl<'a, F, T> Extrapolator<'a, F, CSCArray<'a, F>, T, Ix2> for Solver
where
    F: Float,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &'a DatasetBase<CSCArray<'a, F>, T>,
        XW_acc: &'a mut ArrayBase<OwnedRepr<F>, Ix2>,
        W_acc: ArrayBase<ViewRepr<&'a F>, Ix2>,
        ws: ArrayView1<'a, usize>,
    ) {
        let X = dataset.design_matrix();
        let n_tasks = dataset.n_tasks();
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

/// This implementation block implements the working set construction methods
/// for single-task solvers.
///

impl<'a, F, DF, P, DM, T> WorkingSet<'a, F, DF, P, DM, T, Ix1> for Solver
where
    F: Float,
    DF: Datafit<'a, F, DM, T, Ix1, Output = F>,
    P: Penalty<'a, F, Ix1, Input = F, Output = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn kkt_violation(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        w: ArrayBase<ViewRepr<&'a F>, Ix1>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
        ws: ArrayView1<'a, usize>,
        datafit: &'a DF,
        penalty: &'a P,
    ) -> (Array1<F>, F) {
        // Contruct grad restricted to working set
        let ws_size = ws.len();
        let mut grad_ws = Array1::<F>::zeros(ws_size);
        for (idx, &j) in ws.iter().enumerate() {
            grad_ws[idx] = datafit.gradient_j(dataset, Xw, j);
        }
        // Compute KKT
        let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
        (kkt_ws, kkt_ws_max)
    }

    fn construct_ws_from_kkt(
        &self,
        kkt: &'a mut Array1<F>,
        w: ArrayBase<ViewRepr<&'a F>, Ix1>,
        p0: usize,
    ) -> (Array1<usize>, usize) {
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
}

/// This implementation block implements the working set construction methods
/// for multi-task solvers.
///
impl<'a, F, DF, P, DM, T> WorkingSet<'a, F, DF, P, DM, T, Ix2> for Solver
where
    F: Float,
    DF: Datafit<'a, F, DM, T, Ix2, Output = ArrayBase<OwnedRepr<F>, Ix1>>,
    P: Penalty<
        'a,
        F,
        Ix2,
        Input = ArrayBase<ViewRepr<&'a F>, Ix1>,
        Output = ArrayBase<OwnedRepr<F>, Ix1>,
    >,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn kkt_violation(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        w: ArrayBase<ViewRepr<&'a F>, Ix2>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix2>,
        ws: ArrayView1<'a, usize>,
        datafit: &'a DF,
        penalty: &'a P,
    ) -> (Array1<F>, F) {
        // Compute gradient restricted to working set
        let ws_size = ws.len();
        let n_tasks = dataset.n_tasks();
        let mut grad_ws = Array2::<F>::zeros((ws_size, n_tasks));
        for (idx, &j) in ws.iter().enumerate() {
            let grad_j = datafit.gradient_j(&dataset, Xw, j);
            for t in 0..n_tasks {
                grad_ws[[idx, t]] = grad_j[t];
            }
        }
        // Compute KKT
        let (kkt_ws, kkt_ws_max) = penalty.subdiff_distance(w, grad_ws.view(), ws);
        (kkt_ws, kkt_ws_max)
    }

    fn construct_ws_from_kkt(
        &self,
        kkt: &'a mut Array1<F>,
        w: ArrayBase<ViewRepr<&'a F>, Ix2>,
        p0: usize,
    ) -> (Array1<usize>, usize) {
        let n_features = w.shape()[0];
        let mut nnz_features: usize = 0;

        for j in 0..n_features {
            if w.slice(s![j, ..]).map(|&x| x.abs()).sum() != F::zero() {
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
}
