use ndarray::{s, Array1, ArrayBase, ArrayView1, Axis, Data, Ix2};

use super::Float;
use crate::datasets::{csc_array::CSCArray, AsSingleTargets, DatasetBase, DesignMatrix};

#[cfg(test)]
mod tests;

/// This trait provides three main methods [`Datafit::initialize`],
/// [`Datafit::value`] and [`Datafit::gradient_j`] to compute useful quantities
/// during the optimization routine.
pub trait Datafit<F: Float, DM: DesignMatrix<Elem = F>, T: AsSingleTargets<Elem = F>> {
    /// This method is called before looping onto the features, to precompute
    /// the Lipschitz constants (used as stepsizes) and the matrix-vector
    /// product XTy.
    fn initialize(&mut self, dataset: &DatasetBase<DM, T>);

    /// This method is called when evaluating the objective value.
    ///
    /// It is jointly used with [`Penalty::value`] in order to compute the
    /// value of the objective.
    fn value(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>) -> F;

    /// This method computes the gradient of the datafit with respect to the
    /// weight vector.
    fn gradient_j(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>, j: usize) -> F;

    /// This method computes the full gradient by calling
    /// [`Datafit::gradient_j`].
    fn full_grad(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>) -> Array1<F>;

    /// A getter method for the pre-computed Lipschitz constants.
    fn lipschitz(&self) -> ArrayView1<F>;

    /// A getter method for the matrix-vector product XTy.
    fn Xty(&self) -> ArrayView1<F>;
}

/// Quadratic datafit
///
/// The squared-norm residuals datafit used in most regression settings.
/// Conjointly used with penalties implementing the [`Penalty`] trait, it allows
/// to create a wide variety of regression models (e.g. LASSO, MCP Regressor,
/// etc.). It stores the pre-computed quantities useful during the optimization
/// routine.
#[derive(Debug, Clone, PartialEq)]
pub struct Quadratic<F: Float> {
    lipschitz: Array1<F>,
    Xty: Array1<F>,
}

impl<F: Float> Default for Quadratic<F> {
    fn default() -> Quadratic<F> {
        Quadratic {
            lipschitz: Array1::<F>::zeros(1),
            Xty: Array1::<F>::zeros(1),
        }
    }
}

impl<F, DM, T, DF> Datafit<F, DM, T> for &DF
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
{
    fn initialize(&mut self, dataset: &DatasetBase<DM, T>) {
        (*self).initialize(dataset);
    }

    fn gradient_j(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>, j: usize) -> F {
        (*self).gradient_j(dataset, Xw, j)
    }

    fn full_grad(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>) -> Array1<F> {
        (*self).full_grad(dataset, Xw)
    }

    fn value(&self, dataset: &DatasetBase<DM, T>, Xw: ArrayView1<F>) -> F {
        (*self).value(dataset, Xw)
    }

    fn lipschitz(&self) -> ArrayView1<F> {
        (*self).lipschitz()
    }

    fn Xty(&self) -> ArrayView1<F> {
        (*self).Xty()
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsSingleTargets<Elem = F>> Datafit<F, ArrayBase<D, Ix2>, T>
    for Quadratic<F>
{
    /// This method pre-computes the Lipschitz constants and the matrix-vector
    /// product XTy useful during the optimization routine.
    fn initialize(&mut self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) {
        let n_samples = F::cast(dataset.targets().n_samples());
        let X = dataset.design_matrix();
        let y = dataset.targets().try_single_target().unwrap();
        self.Xty = X.t().dot(&y);
        self.lipschitz = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
    }

    /// This method computes the value of the gradient at some point w for
    /// coordinate j.
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

    /// This method computes the full gradient of the datafit with respect to
    /// the weight vector.
    fn full_grad(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
        Xw: ArrayView1<F>,
    ) -> Array1<F> {
        Array1::from_iter(
            (0..dataset.design_matrix().n_features())
                .into_iter()
                .map(|j| self.gradient_j(dataset, Xw, j))
                .collect::<Vec<F>>(),
        )
    }

    /// This method computes the value of the datafit given the model fit.
    fn value(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>, Xw: ArrayView1<F>) -> F {
        let n_samples = dataset.targets().n_samples();
        let y = dataset.targets().try_single_target().unwrap();
        let r = &y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    // A getter method for the Lipschitz constants.
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    // A getter method for Xty.
    fn Xty(&self) -> ArrayView1<F> {
        self.Xty.view()
    }
}

/// This implementation block implements the [`Datafit`] for sparse matrices
/// (CSC arrays). The methods [`Datafit::initialize`] and [`Datafit::gradient_j`]
/// are modified to exploit the sparse structure of the design matrix.
impl<F: Float, T: AsSingleTargets<Elem = F>> Datafit<F, CSCArray<'_, F>, T> for Quadratic<F> {
    /// This method initializes the datafit by pre-computing useful quantities
    /// with sparse matrices.
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

    /// This method computes the value of the gradient at some point w for
    /// coordinate j using sparse matrices.
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

    /// This method computes the gradient at some point w using sparse matrices.
    fn full_grad(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>, Xw: ArrayView1<F>) -> Array1<F> {
        Array1::from_iter(
            (0..dataset.design_matrix().n_features())
                .into_iter()
                .map(|j| self.gradient_j(dataset, Xw, j))
                .collect::<Vec<F>>(),
        )
    }

    /// This method computes the value of the datafit for some model fit.
    fn value(&self, dataset: &DatasetBase<CSCArray<'_, F>, T>, Xw: ArrayView1<F>) -> F {
        let n_samples = dataset.targets().n_samples();
        let y = dataset.targets().try_single_target().unwrap();
        let r = &y - &Xw;
        let val = r.dot(&r) / F::cast(2 * n_samples);
        val
    }

    /// A getter method for Lipschitz constants.
    fn lipschitz(&self) -> ArrayView1<F> {
        self.lipschitz.view()
    }

    /// A getter method for the matrix-vector product XTy.
    fn Xty(&self) -> ArrayView1<F> {
        self.Xty.view()
    }
}
