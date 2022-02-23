extern crate ndarray;

use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2, OwnedRepr,
};

use super::Float;
use crate::datasets::{csc_array::CSCArray, AsMultiTargets, DatasetBase, DesignMatrix};

#[cfg(test)]
mod tests;

pub trait MultiTaskDatafit<F, DM, T>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
{
    fn initialize(&mut self, dataset: &DatasetBase<DM, T>);
    fn value(&self, dataset: &DatasetBase<DM, T>, XW: ArrayView2<F>) -> F;
    fn gradient_j(
        &self,
        dataset: &DatasetBase<DM, T>,
        XW: ArrayView2<F>,
        j: usize,
    ) -> ArrayBase<OwnedRepr<F>, Ix1>;
    fn full_grad(
        &self,
        dataset: &DatasetBase<DM, T>,
        XW: ArrayView2<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2>;

    fn lipschitz(&self) -> ArrayView1<F>;
    fn XtY(&self) -> ArrayView2<F>;
}

/// Multi-Task Quadratic datafit
///

pub struct QuadraticMultiTask<F: Float> {
    lipschitz: ArrayBase<OwnedRepr<F>, Ix1>,
    XtY: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F: Float> Default for QuadraticMultiTask<F> {
    fn default() -> QuadraticMultiTask<F> {
        QuadraticMultiTask {
            lipschitz: Array1::zeros(1),
            XtY: Array2::zeros((1, 1)),
        }
    }
}

impl<F, D> MultiTaskDatafit<F, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.targets().n_samples());
        let X = dataset.design_matrix();
        let Y = dataset.targets();
        let xty = X.t().dot(Y);
        self.lipschitz = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
        self.XtY = xty;
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> F {
        let n_samples = dataset.targets().n_samples();
        let Y = dataset.targets();
        let R = Y - &XW;
        let frob = R.fold(F::zero(), |sum, &x| sum + x * x);
        frob / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
        j: usize,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_samples = F::cast(dataset.targets().n_samples());
        let X = dataset.design_matrix();
        let Xj: ArrayView1<F> = X.slice(s![.., j]);
        let grad = Xj.dot(&XW) - self.XtY.slice(s![j, ..]);
        grad / n_samples
    }

    /// Computes the value of the gradient at some point w
    fn full_grad(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_features = dataset.design_matrix().n_features();
        let n_tasks = dataset.targets().n_tasks();
        let mut grad = Array2::<F>::zeros((n_features, n_tasks));
        for j in 0..n_features {
            grad.slice_mut(s![j, ..])
                .assign(&self.gradient_j(dataset, XW, j));
        }
        grad
    }

    // Getter for Lipschitz
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn XtY(&self) -> ArrayView2<F> {
        self.XtY.view()
    }
}

impl<F, D> MultiTaskDatafit<F, CSCArray<'_, F>, ArrayBase<D, Ix2>> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.targets().n_samples());
        let n_features = dataset.design_matrix().n_features();
        let n_tasks = dataset.targets().n_tasks();

        let X = dataset.design_matrix();
        let Y = dataset.targets();

        self.XtY = Array2::<F>::zeros((n_features, n_tasks));
        self.lipschitz = Array1::<F>::zeros(n_features);

        for j in 0..n_features {
            let mut nrm2 = F::zero();
            let mut xty = Array1::<F>::zeros(n_tasks);
            for idx in X.indptr[j]..X.indptr[j + 1] {
                nrm2 += X.data[idx as usize] * X.data[idx as usize];
                for t in 0..n_tasks {
                    xty[t] += X.data[idx as usize] * Y[[X.indices[idx as usize] as usize, t]];
                }
            }
            self.lipschitz[j] = nrm2 / n_samples;
            self.XtY.slice_mut(s![j, ..]).assign(&xty);
        }
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> F {
        let n_samples = dataset.targets().n_samples();
        let Y = dataset.targets();
        let R = Y - &XW;
        let frob = R.fold(F::zero(), |sum, &x| sum + x * x);
        frob / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
        j: usize,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_samples = F::cast(dataset.targets().n_samples());
        let n_tasks = dataset.targets().n_tasks();
        let X = dataset.design_matrix();
        let mut XjTXW = Array1::<F>::zeros(n_tasks);
        for i in X.indptr[j]..X.indptr[j + 1] {
            for t in 0..n_tasks {
                XjTXW[t] += X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
            }
        }
        let grad_j = XjTXW - self.XtY.slice(s![j, ..]);
        grad_j / n_samples
    }

    /// Computes the value of the gradient at some point w
    fn full_grad(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_tasks = dataset.targets().n_tasks();
        let n_features = dataset.design_matrix().n_features();
        let mut grad = Array2::<F>::zeros((n_features, n_tasks));
        for j in 0..n_features {
            grad.slice_mut(s![j, ..])
                .assign(&self.gradient_j(dataset, XW, j));
        }
        grad
    }

    // Getter for Lipschitz
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn XtY(&self) -> ArrayView2<F> {
        self.XtY.view()
    }
}
