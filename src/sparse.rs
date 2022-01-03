#[derive(Debug)]
pub struct CSRArray<T: Float> {
    data: Vec<T>,
    indices: Vec<usize>,
    indptr: Vec<usize>,
}
