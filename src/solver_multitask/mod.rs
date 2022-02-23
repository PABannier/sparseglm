extern crate ndarray;

use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};

use super::Float;
use crate::datafits_multitask::MultiTaskDatafit;
use crate::datasets::{csc_array::CSCArray, AsMultiTargets, DatasetBase, DesignMatrix};
use crate::penalties_multitask::PenaltyMultiTask;

#[cfg(test)]
mod tests;

pub struct MultiTaskSolver {}

pub trait BCDSolver<F, DF, P, DM, T>
where
    F: Float,
    DF: MultiTaskDatafit<F, DM, T>,
    P: PenaltyMultiTask<F>,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
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

pub trait MultiTaskExtrapolator<F: Float, DM: DesignMatrix<Elem = F>, T: AsMultiTargets<Elem = F>> {
    fn extrapolate(
        &self,
        dataset: &DatasetBase<DM, T>,
        XW_acc: &mut Array2<F>,
        W_acc: ArrayView2<F>,
        ws: ArrayView1<usize>,
    );
}

/// This implementation block implements the coordinate descent epoch for dense
/// design matrices.

impl<F, D, DF, P, T> BCDSolver<F, DF, P, ArrayBase<D, Ix2>, T> for MultiTaskSolver
where
    F: Float,
    D: Data<Elem = F>,
    DF: MultiTaskDatafit<F, ArrayBase<D, Ix2>, T>,
    P: PenaltyMultiTask<F>,
    T: AsMultiTargets<Elem = F>,
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
        let n_samples = dataset.targets().n_samples();
        let n_tasks = dataset.targets().n_tasks();

        let X = dataset.design_matrix();
        let lipschitz = datafit.lipschitz();

        for &j in ws {
            if lipschitz[j] == F::zero() {
                continue;
            }
            let Xj: ArrayView1<F> = X.slice(s![.., j]);
            let old_W_j = W.slice(s![j, ..]).to_owned();
            let grad_j = datafit.gradient_j(dataset, XW.view(), j);

            let step = &old_W_j - grad_j / lipschitz[j];
            let upd = penalty.prox_op(step.view(), F::one() / lipschitz[j]);
            W.slice_mut(s![j, ..]).assign(&upd);

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

impl<'a, F, DF, P, T> BCDSolver<F, DF, P, CSCArray<'a, F>, T> for MultiTaskSolver
where
    F: Float,
    DF: MultiTaskDatafit<F, CSCArray<'a, F>, T>,
    P: PenaltyMultiTask<F>,
    T: AsMultiTargets<Elem = F>,
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
        let n_tasks = dataset.targets().n_tasks();

        let X = dataset.design_matrix();
        let lipschitz = datafit.lipschitz();

        for &j in ws {
            if lipschitz[j] == F::zero() {
                continue;
            }
            let old_W_j = W.slice(s![j, ..]).to_owned();
            let grad_j = datafit.gradient_j(dataset, XW.view(), j);

            let step = &old_W_j - grad_j / lipschitz[j];
            let upd = penalty.prox_op(step.view(), F::one() / lipschitz[j]);
            W.slice_mut(s![j, ..]).assign(&upd);

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
impl<F, D, T> MultiTaskExtrapolator<F, ArrayBase<D, Ix2>, T> for MultiTaskSolver
where
    F: Float,
    D: Data<Elem = F>,
    T: AsMultiTargets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        XW_acc: &mut Array2<F>,
        W_acc: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) {
        let X = dataset.design_matrix();
        let n_samples = dataset.targets().n_samples();
        let n_tasks = dataset.targets().n_tasks();
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
impl<'a, F, T> MultiTaskExtrapolator<F, CSCArray<'a, F>, T> for MultiTaskSolver
where
    F: Float,
    T: AsMultiTargets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        XW_acc: &mut Array2<F>,
        W_acc: ArrayView2<F>,
        ws: ArrayView1<usize>,
    ) {
        let X = dataset.design_matrix();
        let n_tasks = dataset.targets().n_tasks();
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
