extern crate ndarray;
extern crate num;

use ndarray::{ArrayView1, ArrayView2};
use num::Float;

#[derive(Debug)]
pub struct CSCArray<'a, T: Float> {
    pub data: ArrayView1<'a, T>,
    pub indices: ArrayView1<'a, i32>,
    pub indptr: ArrayView1<'a, i32>,
}

impl<'a, T: Float> CSCArray<'a, T> {
    pub fn new(
        data: ArrayView1<'a, T>,
        indices: ArrayView1<'a, i32>,
        indptr: ArrayView1<'a, i32>,
    ) -> CSCArray<'a, T> {
        CSCArray {
            data,
            indices,
            indptr,
        }
    }
}

#[derive(Clone, Copy)]
pub enum MatrixParam<'a, T: Float> {
    DenseMatrix(ArrayView2<'a, T>),
    SparseMatrix(&'a CSCArray<'a, T>),
}
