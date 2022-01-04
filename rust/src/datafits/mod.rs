extern crate ndarray;
extern crate num;

use ndarray::{s, Array1, ArrayView1, ArrayView2, Axis};
use num::Float;

#[cfg(test)]
mod tests;

pub trait Datafit<T: Float> {
    fn initialize(&mut self, X: ArrayView2<T>, y: ArrayView1<T>);
    fn value(&self, y: ArrayView1<T>, w: ArrayView1<T>, Xw: ArrayView1<T>) -> T;
    fn gradient_scalar(
        &self,
        X: ArrayView2<T>,
        y: ArrayView1<T>,
        w: ArrayView1<T>,
        Xw: ArrayView1<T>,
        j: usize,
    ) -> T;

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
    /// Computes the value of the datafit
    fn value(&self, y: ArrayView1<T>, _w: ArrayView1<T>, Xw: ArrayView1<T>) -> T {
        let r = &y - &Xw;
        let denom = T::from(2 * y.len()).unwrap();
        let val = r.dot(&r) / denom;
        val
    }
    /// Computes the value of the gradient at some point w
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
    // Getter for Lipschitz
    fn get_lipschitz(&self) -> ArrayView1<T> {
        self.lipschitz.view()
    }
    // Getter for Xty
    fn get_Xty(&self) -> ArrayView1<T> {
        self.Xty.view()
    }
}