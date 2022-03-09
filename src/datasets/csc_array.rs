use ndarray::ArrayView1;

/// CSCArrays allows to optimally store very sparse matrices in a column-first
/// fashion. For more information, see:
/// `https://scipy-lectures.org/advanced/scipy_sparse/csc_matrix.html`
#[derive(Debug, Clone, PartialEq)]
pub struct CSCArray<'a, T> {
    pub data: ArrayView1<'a, T>,
    pub indices: ArrayView1<'a, i32>,
    pub indptr: ArrayView1<'a, i32>,
}

impl<'a, T> CSCArray<'a, T> {
    /// This instantiates a [`CSCArray`].
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
