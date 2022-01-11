extern crate ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::{
    s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2, OwnedRepr,
};

use super::Float;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};

#[cfg(test)]
mod tests;

pub trait MultiTaskDatafit<F, D, DM, T>
where
    F: Float,
    D: Data<Elem = F>,
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
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

pub struct QuadraticMultiTask<F>
where
    F: Float,
{
    lipschitz: ArrayBase<OwnedRepr<F>, Ix1>,
    XtY: ArrayBase<OwnedRepr<F>, Ix2>,
}

impl<F> Default for QuadraticMultiTask<F>
where
    F: Float,
{
    fn default() -> QuadraticMultiTask<F> {
        QuadraticMultiTask {
            lipschitz: Array1::zeros(1),
            XtY: Array2::zeros((1, 1)),
        }
    }
}

impl<F, D> MultiTaskDatafit<F, D, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix;
        let Y = dataset.targets;

        let mut xty = Array2::<F>::zeros((n_features, n_tasks));
        general_mat_mul(F::one(), &X.t(), &Y, F::one(), &mut xty);

        let lc = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
        self.lipschitz = lc;
        self.XtY = xty;
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> F {
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();
        let Y = dataset.targets;

        let R = &Y - &XW;
        let mut val = F::zero();
        for i in 0..n_samples {
            for j in 0..n_tasks {
                val = val + R[[i, j]] * R[[i, j]];
            }
        }
        val / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
        j: usize,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_samples = F::cast(dataset.n_samples());
        let n_tasks = dataset.n_tasks();
        let X = dataset.design_matrix;

        let Xj: ArrayView1<F> = X.slice(s![.., j]);
        let mut grad = Xj.dot(&XW) - self.XtY.slice(s![j, ..]);
        for t in 0..n_tasks {
            grad[t] /= n_samples;
        }
        grad
    }

    /// Computes the value of the gradient at some point w
    fn full_grad(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();
        let X = dataset.design_matrix;

        let mut grad = Array2::<F>::zeros((n_features, n_tasks));

        for j in 0..n_features {
            let Xj: ArrayView1<F> = X.slice(s![.., j]);
            let mut grad_j = Xj.dot(&XW) - self.XtY.slice(s![j, ..]);
            for t in 0..n_tasks {
                grad_j[t] /= n_samples;
            }

            // Assign
            for t in 0..n_tasks {
                grad[[j, t]] = grad_j[t];
            }
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

impl<F, D> MultiTaskDatafit<F, D, CSCArray<'_, F>, ArrayBase<D, Ix2>> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix;
        let Y = dataset.targets;

        self.XtY = Array2::<F>::zeros((n_features, n_tasks));
        self.lipschitz = Array1::<F>::zeros(n_features);

        for j in 0..n_features {
            let mut nrm2 = F::zero();
            let mut xty = Array1::<F>::zeros(n_tasks);
            for idx in X.indptr[j]..X.indptr[j + 1] {
                nrm2 = nrm2 + X.data[idx as usize] * X.data[idx as usize];
                for t in 0..n_tasks {
                    xty[t] =
                        xty[t] + X.data[idx as usize] * Y[[X.indices[idx as usize] as usize, t]];
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
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();
        let Y = dataset.targets;

        let R = &Y - &XW;
        let mut val = F::zero();
        for i in 0..n_samples {
            for j in 0..n_tasks {
                val = val + R[[i, j]] * R[[i, j]];
            }
        }
        val / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
        j: usize,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_samples = F::cast(dataset.n_samples());
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix;
        let mut XjTXW = Array1::<F>::zeros(n_tasks);

        for i in X.indptr[j]..X.indptr[j + 1] {
            for t in 0..n_tasks {
                XjTXW[t] = XjTXW[t] + X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
            }
        }
        let mut grad_j = XjTXW - self.XtY.slice(s![j, ..]);
        for t in 0..n_tasks {
            grad_j[t] = grad_j[t] / n_samples;
        }
        grad_j
    }

    /// Computes the value of the gradient at some point w
    fn full_grad(
        &self,
        dataset: &DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayView2<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix;

        let mut grad = Array2::<F>::zeros((n_features, n_tasks));

        for j in 0..n_features {
            let mut XjTXW = Array1::<F>::zeros(n_tasks);
            for i in X.indptr[j]..X.indptr[j + 1] {
                for t in 0..n_tasks {
                    XjTXW[t] =
                        XjTXW[t] + X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
                }
            }
            let mut grad_j = XjTXW - self.XtY.slice(s![j, ..]);
            for t in 0..n_tasks {
                grad_j[t] /= n_samples;
            }

            // Assign
            for t in 0..n_tasks {
                grad[[j, t]] = grad_j[t];
            }
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
