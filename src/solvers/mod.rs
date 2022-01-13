extern crate ndarray;

use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, Data, Dimension, Ix1, Ix2, OwnedRepr, ViewRepr,
};

use super::Float;
use crate::datafits::Datafit;
use crate::datafits_multitask::MultiTaskDatafit;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};
use crate::penalties::Penalty;
use crate::penalties_multitask::PenaltyMultiTask;

#[cfg(test)]
mod tests;

pub struct Solver {}

impl Solver {
    pub fn new() -> Solver {
        Solver {}
    }
}

pub trait CDSolver<F, DF, P, DM, T>
where
    F: Float,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &DF,
        penalty: &P,
        w: &mut Array1<F>,
        Xw: &mut Array1<F>,
        ws: ArrayView1<usize>,
    );
}

pub trait Extrapolator<F, DM, T, I>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
    I: Dimension,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<DM, T>,
        Xw_acc: &mut ArrayBase<OwnedRepr<F>, I>,
        w_acc: ArrayBase<ViewRepr<&F>, I>,
        ws: ArrayView1<usize>,
    );
}

/// This implementation block implements the coordinate descent epoch for dense
/// design matrices.

impl<F, D, DF, P, T> CDSolver<F, DF, P, ArrayBase<D, Ix2>, T> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    DF: Datafit<F, ArrayBase<D, Ix2>, T>,
    P: Penalty<F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        datafit: &DF,
        penalty: &P,
        w: &mut Array1<F>,
        Xw: &mut Array1<F>,
        ws: ArrayView1<usize>,
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

impl<'a, F, DF, P, T> CDSolver<F, DF, P, CSCArray<'a, F>, T> for Solver
where
    F: Float,
    DF: Datafit<F, CSCArray<'a, F>, T>,
    P: Penalty<F>,
    T: Targets<Elem = F>,
{
    fn cd_epoch(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        datafit: &DF,
        penalty: &P,
        w: &mut Array1<F>,
        Xw: &mut Array1<F>,
        ws: ArrayView1<usize>,
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
impl<F, D, T> Extrapolator<F, ArrayBase<D, Ix2>, T, Ix1> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw_acc: &mut ArrayBase<OwnedRepr<F>, Ix1>,
        w_acc: ArrayBase<ViewRepr<&F>, Ix1>,
        ws: ArrayView1<usize>,
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
impl<'a, F, T> Extrapolator<F, CSCArray<'a, F>, T, Ix1> for Solver
where
    F: Float,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        Xw_acc: &mut ArrayBase<OwnedRepr<F>, Ix1>,
        w_acc: ArrayBase<ViewRepr<&F>, Ix1>,
        ws: ArrayView1<usize>,
    ) {
        let X = dataset.design_matrix();
        for &j in ws {
            for idx in X.indptr[j]..X.indptr[j + 1] {
                Xw_acc[X.indices[idx as usize] as usize] += X.data[idx as usize] * w_acc[j];
            }
        }
    }
}

pub trait BCDSolver<F, DF, P, DM, T>
where
    F: Float,
    DF: MultiTaskDatafit<F, DM, T>,
    P: PenaltyMultiTask<F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &DF,
        penalty: &P,
        W: &mut Array2<F>,
        XW: &mut Array2<F>,
        ws: ArrayView1<usize>,
    );
}

/// This implementation block implements the coordinate descent epoch for dense
/// design matrices.

impl<F, D, DF, P, T> BCDSolver<F, DF, P, ArrayBase<D, Ix2>, T> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    DF: MultiTaskDatafit<F, ArrayBase<D, Ix2>, T>,
    P: PenaltyMultiTask<F>,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        datafit: &DF,
        penalty: &P,
        W: &mut Array2<F>,
        XW: &mut Array2<F>,
        ws: ArrayView1<usize>,
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

impl<'a, F, DF, P, T> BCDSolver<F, DF, P, CSCArray<'a, F>, T> for Solver
where
    F: Float,
    DF: MultiTaskDatafit<F, CSCArray<'a, F>, T>,
    P: PenaltyMultiTask<F>,
    T: Targets<Elem = F>,
{
    fn bcd_epoch(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        datafit: &DF,
        penalty: &P,
        W: &mut Array2<F>,
        XW: &mut Array2<F>,
        ws: ArrayView1<usize>,
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
impl<F, D, T> Extrapolator<F, ArrayBase<D, Ix2>, T, Ix2> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        XW_acc: &mut ArrayBase<OwnedRepr<F>, Ix2>,
        W_acc: ArrayBase<ViewRepr<&F>, Ix2>,
        ws: ArrayView1<usize>,
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
impl<'a, F, T> Extrapolator<F, CSCArray<'a, F>, T, Ix2> for Solver
where
    F: Float,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        XW_acc: &mut ArrayBase<OwnedRepr<F>, Ix2>,
        W_acc: ArrayBase<ViewRepr<&F>, Ix2>,
        ws: ArrayView1<usize>,
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
