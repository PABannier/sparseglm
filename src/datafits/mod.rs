extern crate ndarray;

use ndarray::{s, Array1, ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2, OwnedRepr};

use super::Float;
use crate::datasets::{csc_array::CSCArray, AsSingleTargets, DatasetBase, DesignMatrix};

#[cfg(test)]
mod tests;

pub trait Datafit<F: Float, DM: DesignMatrix<Elem = F>, T: AsSingleTargets<Elem = F>> {
    fn initialize(&mut self, dataset: &DatasetBase<DM, T>);
    fn value(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>) -> F;
    fn gradient_j(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>, j: usize) -> F;
    fn full_grad(
        &self,
        dataset: &DatasetBase<DM, T>,
        Xw: ArrayView1<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix1>;

    fn lipschitz(&self) -> ArrayView1<F>;
    fn Xty(&self) -> ArrayView1<F>;
}

/// Quadratic datafit
///

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

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>> Datafit<F, ArrayBase<D, Ix2>, T>
    for Quadratic<F>
{
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) {
        let n_samples = F::cast(dataset.targets().n_samples());
        let X = dataset.design_matrix();
        let y = dataset.targets().try_single_target().unwrap();
        self.Xty = X.t().dot(&y);
        self.lipschitz = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw: ArrayView1<F>,
        j: usize,
    ) -> F {
        let n_samples = dataset.targets().n_samples();
        let X = dataset.design_matrix();
        let _res = X.slice(s![.., j]).dot(&Xw);
        (_res - self.Xty[j]) / F::cast(n_samples)
    }

    /// Compute the gradient at some point w
    fn full_grad(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw: ArrayView1<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_features = dataset.design_matrix().n_features();
        let mut grad = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            grad[j] = self.gradient_j(dataset, Xw, j);
        }
        grad
    }

    /// Computes the value of the datafit
    fn value(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, Xw: ArrayView1<F>) -> F {
        let n_samples = dataset.targets().n_samples();
        let y = dataset.targets().try_single_target().unwrap();
        let r = &y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    // Getter for Lipschitz constants
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayView1<F> {
        self.Xty.view()
    }
}

impl<F: Float, T: AsSingleTargets<Elem = F>> Datafit<F, CSCArray<'_, F>, T> for Quadratic<F> {
    /// Initializes the datafit by pre-computing useful quantities with sparse matrices
    fn initialize(&mut self, dataset: &DatasetBase<CSCArray<'_, F>, T>) {
        let n_samples = dataset.targets().n_samples();
        let n_features = dataset.design_matrix().n_features();
        let X = dataset.design_matrix();
        let y = dataset.targets().try_single_target().unwrap();
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
        dataset: &DatasetBase<CSCArray<'_, F>, T>,
        Xw: ArrayView1<F>,
        j: usize,
    ) -> F {
        let n_samples = dataset.targets().n_samples();
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
        dataset: &DatasetBase<CSCArray<'_, F>, T>,
        Xw: ArrayView1<F>,
    ) -> ArrayBase<OwnedRepr<F>, Ix1> {
        let n_features = dataset.design_matrix().n_features();
        let mut grad = Array1::<F>::zeros(n_features);
        for j in 0..n_features {
            grad[j] = self.gradient_j(dataset, Xw, j);
        }
        grad
    }

    /// Computes the value of the datafit
    fn value(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>, Xw: ArrayView1<F>) -> F {
        let n_samples = dataset.targets().n_samples();
        let y = dataset.targets().try_single_target().unwrap();
        let r = &y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    // Getter for Lipschitz constants
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn Xty(&self) -> ArrayView1<F> {
        self.Xty.view()
    }
}
