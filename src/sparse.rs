extern crate num;

use num::Float;

#[derive(Debug)]
pub struct CSRArray<T: Float> {
    pub data: Vec<T>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}
