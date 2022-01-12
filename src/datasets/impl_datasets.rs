use super::{csc_array::CSCArray, DatasetBase};

use ndarray::{ArrayBase, Data, Dimension, Ix2};

/// Implementation without constraints on records and targets
///
/// This implementation block provides a method for the creation of datasets
/// from dense matrices.
impl<F, D, I> From<(ArrayBase<D, Ix2>, ArrayBase<D, I>)>
    for DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, I>>
where
    D: Data<Elem = F>,
    I: Dimension,
{
    fn from(data: (ArrayBase<D, Ix2>, ArrayBase<D, I>)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}

/// This implementation block provides a method for the creation of datasets
/// from sparse matrices.
impl<'a, F, D, I> From<(CSCArray<'a, F>, ArrayBase<D, I>)>
    for DatasetBase<CSCArray<'a, F>, ArrayBase<D, I>>
where
    D: Data<Elem = F>,
    I: Dimension,
{
    fn from(data: (CSCArray<'a, F>, ArrayBase<D, I>)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}
