extern crate ndarray;
extern crate num;

use ndarray::{s, Array1, ArrayView1, ArrayView2, Axis};
use num::Float;

use crate::sparse::CSCArray;

#[cfg(test)]
mod tests;

pub trait Datafit<T: Float> {
    fn initialize(&mut self, X: ArrayView2<T>, y: ArrayView1<T>);
    fn initialize_sparse(&mut self, X: &CSCArray<T>, y: ArrayView1<T>);
    fn value(&self, y: ArrayView1<T>, w: ArrayView1<T>, Xw: ArrayView1<T>) -> T;

    fn gradient_scalar(
        &self,
        X: ArrayView2<T>,
        y: ArrayView1<T>,
        w: ArrayView1<T>,
        Xw: ArrayView1<T>,
        j: usize,
    ) -> T;

    fn gradient_scalar_sparse(
        &self,
        X: &CSCArray<T>,
        y: ArrayView1<T>,
        Xw: ArrayView1<T>,
        j: usize,
    ) -> T;

    fn full_grad_sparse(&self, X: &CSCArray<T>, y: ArrayView1<T>, Xw: ArrayView1<T>) -> Array1<T>;
    fn get_lipschitz(&self) -> ArrayView1<T>;
    fn get_Xty(&self) -> ArrayView1<T>;
}

/// Quadratic datafit
///

pub struct Quadratic<T: Float> {
    lipschitz: Array1<T>,
    Xty: Array1<T>,
}

impl<T: Float> Default for Quadratic<T> {
    fn default() -> Quadratic<T> {
        Quadratic {
            lipschitz: Array1::zeros(1),
            Xty: Array1::zeros(1),
        }
    }
}

impl<'a, T: 'static + Float> Datafit<T> for Quadratic<T> {
    /// Initializes the datafit by pre-computing useful quantities
    fn initialize(&mut self, X: ArrayView2<T>, y: ArrayView1<T>) {
        self.Xty = X.t().dot(&y);
        let n_samples = T::from(X.shape()[0]).unwrap();
        let lc = X.map_axis(Axis(0), |Xj| Xj.dot(&Xj) / n_samples);
        self.lipschitz = lc;
    }
    /// Initializes the datafit by pre-computing useful quantites with sparse matrices
    fn initialize_sparse(&mut self, X: &CSCArray<T>, y: ArrayView1<T>) {
        let n_features = X.indptr.len() - 1;
        self.Xty = Array1::<T>::zeros(n_features);
        self.lipschitz = Array1::<T>::zeros(n_features);
        for j in 0..n_features {
            let mut nrm2 = T::zero();
            let mut xty = T::zero();
            for idx in X.indptr[j]..X.indptr[j + 1] {
                nrm2 = nrm2 + X.data[idx] * X.data[idx];
                xty = xty + X.data[idx] * y[X.indices[idx]];
            }
            self.lipschitz[j] = nrm2 / T::from(y.len()).unwrap();
            self.Xty[j] = xty;
        }
    }

    /// Computes the value of the datafit
    fn value(&self, y: ArrayView1<T>, _w: ArrayView1<T>, Xw: ArrayView1<T>) -> T {
        let r = &y - &Xw;
        let denom = T::from(2 * y.len()).unwrap();
        let val = r.dot(&r) / denom;
        val
    }

    /// Computes the value of the gradient at some point w for coordinate j

    fn gradient_scalar(
        &self,
        X: ArrayView2<T>,
        _y: ArrayView1<T>,
        _w: ArrayView1<T>,
        Xw: ArrayView1<T>,
        j: usize,
    ) -> T {
        let n_samples = T::from(Xw.len()).unwrap();
        let Xj: ArrayView1<T> = X.slice(s![.., j]);
        (Xj.dot(&Xw) - self.Xty[j]) / n_samples
    }

    /// Computes the value of the gradient at some point w for coordinate j using sparse matrices

    fn gradient_scalar_sparse(
        &self,
        X: &CSCArray<T>,
        _y: ArrayView1<T>,
        Xw: ArrayView1<T>,
        j: usize,
    ) -> T {
        let mut XjTXw = T::zero();
        for i in X.indptr[j]..X.indptr[j + 1] {
            XjTXw = XjTXw + X.data[i] * Xw[X.indices[i]];
        }
        return (XjTXw - self.Xty[j]) / T::from(Xw.len()).unwrap();
    }

    /// Computes the gradient at some point w using sparse matrices

    fn full_grad_sparse(&self, X: &CSCArray<T>, y: ArrayView1<T>, Xw: ArrayView1<T>) -> Array1<T> {
        let n_features = X.indptr.len() - 1;
        let n_samples = y.len();
        let mut grad = Array1::<T>::zeros(n_features);
        for j in 0..n_features {
            let mut XjTXw = T::zero();
            for i in X.indptr[j]..X.indptr[j + 1] {
                XjTXw = XjTXw + X.data[i] * Xw[X.indices[i]];
            }
            grad[j] = (XjTXw - self.Xty[j]) / T::from(n_samples).unwrap();
        }
        grad
    }

    // Getter for Lipschitz
    fn get_lipschitz(&self) -> ArrayView1<T> {
        self.lipschitz.view()
    }

    // Getter for Xty
    fn get_Xty(&self) -> ArrayView1<T> {
        self.Xty.view()
    }
}
