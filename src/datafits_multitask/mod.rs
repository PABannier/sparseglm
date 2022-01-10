extern crate ndarray;

use ndarray::linalg::general_mat_mul;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};

use super::Float;
use crate::dataset::CSCArray;

#[cfg(test)]
mod tests;

pub trait DatafitMultiTask<T: Float> {
    fn initialize(&mut self, X: ArrayView2<T>, Y: ArrayView2<T>);
    fn initialize_sparse(&mut self, X: &CSCArray<T>, Y: ArrayView2<T>);
    fn value(&self, Y: ArrayView2<T>, XW: ArrayView2<T>) -> T;
    fn gradient_j(&self, X: ArrayView2<T>, XW: ArrayView2<T>, j: usize) -> Array1<T>;
    fn gradient_j_sparse(&self, X: &CSCArray<T>, XW: ArrayView2<T>, j: usize) -> Array1<T>;
    fn full_grad_sparse(&self, X: &CSCArray<T>, Y: ArrayView2<T>, XW: ArrayView2<T>) -> Array2<T>;

    fn lipschitz(&self) -> ArrayView1<T>;
    fn XtY(&self) -> ArrayView2<T>;
}

/// Quadratic datafit
///

pub struct QuadraticMultiTask<T: Float> {
    lipschitz: Array1<T>,
    XtY: Array2<T>,
}

impl<T: Float> Default for QuadraticMultiTask<T> {
    fn default() -> QuadraticMultiTask<T> {
        QuadraticMultiTask {
            lipschitz: Array1::zeros(1),
            XtY: Array2::zeros((1, 1)),
        }
    }
}

impl<'a, T: 'static + Float> DatafitMultiTask<T> for QuadraticMultiTask<T> {
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, X: ArrayView2<T>, Y: ArrayView2<T>) {
        let n_samples = T::cast(X.shape()[0]);
        let n_features = X.shape()[1];
        let n_tasks = Y.shape()[1];

        let mut xty = Array2::<T>::zeros((n_features, n_tasks));
        general_mat_mul(T::one(), &X.t(), &Y, T::one(), &mut xty);

        let lc = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
        self.lipschitz = lc;
        self.XtY = xty;
    }
    /// Initializes the datafit by pre-computing useful quantites with sparse matrices
    fn initialize_sparse(&mut self, X: &CSCArray<T>, Y: ArrayView2<T>) {
        let n_samples = T::cast(Y.shape()[0]);
        let n_features = X.indptr.len() - 1;
        let n_tasks = Y.shape()[1];

        self.XtY = Array2::<T>::zeros((n_features, n_tasks));
        self.lipschitz = Array1::<T>::zeros(n_features);

        for j in 0..n_features {
            let mut nrm2 = T::zero();
            let mut xty = Array1::<T>::zeros(n_tasks);
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
    fn value(&self, Y: ArrayView2<T>, XW: ArrayView2<T>) -> T {
        let R = &Y - &XW;
        let n_samples = Y.shape()[0];
        let n_tasks = Y.shape()[1];
        let mut val = T::zero();
        for i in 0..n_samples {
            for j in 0..n_tasks {
                val = val + R[[i, j]] * R[[i, j]];
            }
        }
        val / T::cast(2 * n_samples)
    }

    /// Computes the value of the gradient at some point w for coordinate j
    fn gradient_j(&self, X: ArrayView2<T>, XW: ArrayView2<T>, j: usize) -> Array1<T> {
        let n_samples = T::cast(X.shape()[0]);
        let n_tasks = XW.shape()[1];
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        let mut grad = Xj.dot(&XW) - self.XtY.slice(s![j, ..]);
        for t in 0..n_tasks {
            grad[t] = grad[t] / n_samples;
        }
        grad
    }

    /// Computes the value of the gradient at some point w for coordinate j using sparse matrices
    fn gradient_j_sparse(&self, X: &CSCArray<T>, XW: ArrayView2<T>, j: usize) -> Array1<T> {
        let n_samples = T::cast(XW.shape()[0]);
        let n_tasks = XW.shape()[1];
        let mut XjTXW = Array1::<T>::zeros(n_tasks);
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

    /// Computes the gradient at some point w using sparse matrices
    fn full_grad_sparse(&self, X: &CSCArray<T>, Y: ArrayView2<T>, XW: ArrayView2<T>) -> Array2<T> {
        let n_features = X.indptr.len() - 1;
        let n_tasks = XW.shape()[1];
        let n_samples = T::cast(Y.shape()[0]);

        let mut grad = Array2::<T>::zeros((n_features, n_tasks));

        for j in 0..n_features {
            let mut XjTXW = Array1::<T>::zeros(n_tasks);
            for i in X.indptr[j]..X.indptr[j + 1] {
                for t in 0..n_tasks {
                    XjTXW[t] =
                        XjTXW[t] + X.data[i as usize] * XW[[X.indices[i as usize] as usize, t]];
                }
            }
            let mut grad_j = XjTXW - self.XtY.slice(s![j, ..]);
            for t in 0..n_tasks {
                grad_j[t] = grad_j[t] / n_samples;
            }
            grad.slice_mut(s![j, ..]).assign(&grad_j)
        }
        grad
    }

    // Getter for Lipschitz
    fn lipschitz(&self) -> ArrayView1<T> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn XtY(&self) -> ArrayView2<T> {
        self.XtY.view()
    }
}
