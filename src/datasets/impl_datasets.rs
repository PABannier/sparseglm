use super::{csc_array::CSCArray, DatasetBase};

use ndarray::{ArrayBase, Data, Dimension, Ix2};

/// Implementation without constraints on records and targets
///
/// This implementation block provides a method for the creation of datasets
/// from dense matrices.
impl<F, E, D, S, I: Dimension> From<(ArrayBase<D, Ix2>, ArrayBase<S, I>)>
    for DatasetBase<ArrayBase<D, Ix2>, ArrayBase<S, I>>
where
    D: Data<Elem = F>,
    S: Data<Elem = E>,
{
    fn from(data: (ArrayBase<D, Ix2>, ArrayBase<S, I>)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}

/// This implementation block provides a method for the creation of datasets
/// from sparse matrices.
impl<'a, F, E, D, S, I: Dimension> From<(CSCArray<'a, D>, ArrayBase<S, I>)>
    for DatasetBase<CSCArray<'a, D>, ArrayBase<S, I>>
where
    D: Data<Elem = F>,
    S: Data<Elem = E>,
{
    fn from(data: (CSCArray<'a, D>, ArrayBase<S, I>)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}
