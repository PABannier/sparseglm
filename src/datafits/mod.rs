extern crate ndarray;

use ndarray::{
    linalg::general_mat_mul, s, Array1, Array2, ArrayBase, ArrayView1, Axis, Data, Dimension, Ix1,
    Ix2, OwnedRepr, ViewRepr,
};

use super::Float;
use crate::datasets::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};

#[cfg(test)]
mod tests;

pub trait Datafit<'a, F: Float, DM: DesignMatrix, T: Targets, I: Dimension> {
    type Output;

    fn initialize(&mut self, dataset: &'a DatasetBase<DM, T>);
    fn value(&self, dataset: &'a DatasetBase<DM, T>, Xw: ArrayBase<ViewRepr<&'a F>, I>) -> F;
    fn gradient_j(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        Xw: ArrayBase<ViewRepr<&'a F>, I>,
        j: usize,
    ) -> Self::Output;
    fn full_grad(
        &self,
        dataset: &'a DatasetBase<DM, T>,
        Xw: ArrayBase<ViewRepr<&'a F>, I>,
    ) -> ArrayBase<OwnedRepr<F>, I>;

    fn lipschitz(&self) -> ArrayBase<ViewRepr<&F>, Ix1>;
    fn Xty(&self) -> ArrayBase<ViewRepr<&F>, I>;
}

/// Quadratic datafit
pub struct Quadratic<F: Float> {
    lipschitz: ArrayBase<OwnedRepr<F>, Ix1>,
    Xty: ArrayBase<OwnedRepr<F>, Ix1>,
}

impl<F: Float> Default for Quadratic<F> {
    fn default() -> Quadratic<F> {
        Quadratic {
            lipschitz: Array1::<F>::zeros(1),
            Xty: Array1::<F>::zeros(1),
        }
    }
}

impl<'a, F, D> Datafit<'a, F, ArrayBase<D, Ix2>, ArrayBase<D, Ix1>, Ix1> for Quadratic<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Output = F;

    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>) {
        let n_samples = F::cast(dataset.n_samples());
        let X = dataset.design_matrix();
        let y = dataset.targets();
        self.Xty = X.t().dot(y);
        self.lipschitz = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
        j: usize,
    ) -> Self::Output {
        let n_samples = dataset.n_samples();
        let X = dataset.design_matrix();
        let mut _res = F::zero();
        for i in 0..n_samples {
            _res += X[[i, j]] * Xw[i];
        }
        (_res - self.Xty[j]) / F::cast(n_samples)
    }

    /// Compute the gradient at some point w
    fn full_grad(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_features = dataset.n_features();
        let mut grad = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            grad[j] = self.gradient_j(dataset, Xw, j);
        }
        grad
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix1>>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
    ) -> F {
        let n_samples = dataset.n_samples();
        let y = dataset.targets();
        let r = y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    // Getter for Lipschitz constants
    fn lipschitz(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.Xty.view()
    }
}

impl<'a, F, D> Datafit<'a, F, CSCArray<'_, F>, ArrayBase<D, Ix1>, Ix1> for Quadratic<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Output = F;

    /// Initializes the datafit by pre-computing useful quantities with sparse matrices
    fn initialize(&mut self, dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix1>>) {
        let n_samples = dataset.n_samples();
        let n_features = dataset.n_features();
        let X = dataset.design_matrix();
        let y = dataset.targets();
        self.Xty = Array1::<F>::zeros(n_features);
        self.lipschitz = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            let mut nrm2 = F::zero();
            let mut xty = F::zero();
            for idx in X.indptr[j]..X.indptr[j + 1] {
                nrm2 += X.data[idx as usize] * X.data[idx as usize];
                xty += X.data[idx as usize] * y[X.indices[idx as usize] as usize];
            }
            self.lipschitz[j] = nrm2 / F::cast(n_samples);
            self.Xty[j] = xty;
        }
    }

    /// Computes the value of the gradient at some point w for coordinate j using sparse matrices
    fn gradient_j(
        &self,
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix1>>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
        j: usize,
    ) -> Self::Output {
        let n_samples = dataset.n_samples();
        let X = dataset.design_matrix();
        let mut XjTXw = F::zero();
        for i in X.indptr[j]..X.indptr[j + 1] {
            XjTXw += X.data[i as usize] * Xw[X.indices[i as usize] as usize];
        }
        return (XjTXw - self.Xty[j]) / F::cast(n_samples);
    }

    /// Computes the gradient at some point w using sparse matrices
    fn full_grad(
        &self,
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix1>>,
        Xw: ArrayBase<ViewRepr<&'a F>, Ix1>,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_features = dataset.n_features();
        let mut grad = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            grad[j] = self.gradient_j(dataset, Xw, j);
        }
        grad
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix1>>,
        Xw: ArrayView1<'a, F>,
    ) -> F {
        let n_samples = dataset.n_samples();
        let y = dataset.targets();
        let r = y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    // Getter for Lipschitz constants
    fn lipschitz(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.Xty.view()
    }
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

impl<'a, F, D> Datafit<'a, F, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, Ix2> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Output = ArrayBase<OwnedRepr<F>, Ix1>;

    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();
        let Y = dataset.targets();

        let mut xty = Array2::<F>::zeros((n_features, n_tasks));
        general_mat_mul(F::one(), &X.t(), &Y, F::one(), &mut xty);
        self.lipschitz = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
        self.XtY = xty;
    }

    /// Computes the value of the datafit
    fn value(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
    ) -> F {
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();

        let Y = dataset.targets();

        let R = Y - &XW;
        let mut val = F::zero();
        for i in 0..n_samples {
            for j in 0..n_tasks {
                val += R[[i, j]] * R[[i, j]];
            }
        }
        val / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
        j: usize,
    ) -> Self::Output {
        let n_samples = F::cast(dataset.n_samples());
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();

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
        dataset: &'a DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();

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
    fn lipschitz(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.XtY.view()
    }
}

impl<'a, F, D> Datafit<'a, F, CSCArray<'_, F>, ArrayBase<D, Ix2>, Ix2> for QuadraticMultiTask<F>
where
    F: Float,
    D: Data<Elem = F>,
{
    type Output = ArrayBase<OwnedRepr<F>, Ix1>;

    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>) {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

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
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
    ) -> F {
        let n_samples = dataset.n_samples();
        let n_tasks = dataset.n_tasks();

        let Y = dataset.targets();
        let R = Y - &XW;

        let mut val = F::zero();
        for i in 0..n_samples {
            for j in 0..n_tasks {
                val += R[[i, j]] * R[[i, j]];
            }
        }
        val / F::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
        j: usize,
    ) -> Self::Output {
        let n_samples = F::cast(dataset.n_samples());
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();
        let mut XjTXW = Array1::<F>::zeros(n_tasks);

        for i in X.indptr[j]..X.indptr[j + 1] {
            for t in 0..n_tasks {
                XjTXW[t] += X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
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
        dataset: &'a DatasetBase<CSCArray<'_, F>, ArrayBase<D, Ix2>>,
        XW: ArrayBase<ViewRepr<&'a F>, Ix2>,
    ) -> ArrayBase<OwnedRepr<F>, Ix2> {
        let n_samples = F::cast(dataset.n_samples());
        let n_features = dataset.n_features();
        let n_tasks = dataset.n_tasks();

        let X = dataset.design_matrix();

        let mut grad = Array2::<F>::zeros((n_features, n_tasks));

        for j in 0..n_features {
            let mut XjTXW = Array1::<F>::zeros(n_tasks);
            for i in X.indptr[j]..X.indptr[j + 1] {
                for t in 0..n_tasks {
                    XjTXW[t] += X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
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
    fn lipschitz(&self) -> ArrayBase<ViewRepr<&F>, Ix1> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayBase<ViewRepr<&F>, Ix2> {
        self.XtY.view()
    }
}
