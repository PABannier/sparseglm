extern crate num;

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
