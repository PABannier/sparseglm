extern crate ndarray;

use approx::abs_diff_ne;
use ndarray::{s, Array1, ArrayBase, ArrayView1, Data, Ix2};

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};
use crate::penalties::Penalty;

#[cfg(test)]
mod tests;

pub struct Solver {}

pub trait SingleTaskSolver<F, DF, P, DM, T>:
    CDSolver<F, DF, P, DM, T> + Extrapolator<F, DM, T> + FixedPointStrategy<F, DF, P, DM, T>
where
    F: Float,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
}

pub trait FixedPointStrategy<F, DF, P, DM, T>
where
    F: Float,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn dist_fix_point(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        datafit: &DF,
        penalty: &P,
        ws: ArrayView1<usize>,
    ) -> Array1<F>;
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

pub trait Extrapolator<F, DM, T>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<DM, T>,
        Xw_acc: &mut Array1<F>,
        w_acc: ArrayView1<F>,
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
                Xw.scaled_add(w[j] - old_w_j, &X.slice(s![.., j]));
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
impl<F, D, T> Extrapolator<F, ArrayBase<D, Ix2>, T> for Solver
where
    F: Float,
    D: Data<Elem = F>,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw_acc: &mut Array1<F>,
        w_acc: ArrayView1<F>,
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
impl<'a, F, T> Extrapolator<F, CSCArray<'a, F>, T> for Solver
where
    F: Float,
    T: Targets<Elem = F>,
{
    fn extrapolate(
        &self,
        dataset: &DatasetBase<CSCArray<'a, F>, T>,
        Xw_acc: &mut Array1<F>,
        w_acc: ArrayView1<F>,
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

/// This implementation block implements the distance to fixed point strategy
/// to construct working set
///
impl<F, DF, P, DM, T> FixedPointStrategy<F, DF, P, DM, T> for Solver
where
    F: Float,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    fn dist_fix_point(
        &self,
        w: ArrayView1<F>,
        grad: ArrayView1<F>,
        datafit: &DF,
        penalty: &P,
        ws: ArrayView1<usize>,
    ) -> Array1<F> {
        let mut dist_to_fix_point = Array1::<F>::zeros(ws.len());
        for (idx, &j) in ws.iter().enumerate() {
            let lipschitz_j = datafit.lipschitz()[j];
            if abs_diff_ne!(lipschitz_j, F::zero()) {
                dist_to_fix_point[idx] = <F as num_traits::Float>::abs(
                    w[j] - penalty.prox_op(w[j] - grad[idx] / lipschitz_j, F::one() / lipschitz_j),
                );
            }
        }
        dist_to_fix_point
    }
}
