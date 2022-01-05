extern crate ndarray;
extern crate num;

use ndarray::ArrayView2;
use num::Float;

#[derive(Debug)]
pub struct CSCArray<T: Float> {
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

impl<T: Float> CSCArray<T> {
    pub fn new(data: Vec<T>, indices: Vec<usize>, indptr: Vec<usize>) -> CSCArray<T> {
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
    SparseMatrix(&'a CSCArray<T>),
}
