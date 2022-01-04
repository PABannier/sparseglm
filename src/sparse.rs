extern crate num;

use num::Float;

#[derive(Debug)]
pub struct CSRArray<T: Float> {
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

impl<T: Float> CSRArray<T> {
    pub fn new(data: Vec<T>, indices: Vec<usize>, indptr: Vec<usize>) -> CSRArray<T> {
        CSRArray {
            data,
            indices,
            indptr,
        }
    }
}
