use ndarray::ArrayView1;

#[derive(Debug, Clone, PartialEq)]
pub struct CSCArray<'a, T> {
    pub data: ArrayView1<'a, T>,
    pub indices: ArrayView1<'a, i32>,
    pub indptr: ArrayView1<'a, i32>,
}

impl<'a, T> CSCArray<'a, T> {
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
