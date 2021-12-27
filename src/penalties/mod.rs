extern crate ndarray;
extern crate num;

use ndarray::{Array1, ArrayView1, ArrayView2};
use num::{Float, Integer};

pub trait Penalty {
    fn value<T: Float>(self, w: ArrayView1<T>) -> T;
    fn prox_op<T: Float, U: Integer>(self, value: T, stepsize: T, j: U) -> T;
    fn subdiff_distance<T: Float, U: Integer>(
        self,
        w: ArrayView1<T>,
        grad: ArrayView1<T>,
        ws: ArrayView1<U>,
    ) -> Array1<T>;
}
