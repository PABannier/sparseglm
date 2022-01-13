use super::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};

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

/// This implementation block provides methods to get record and target objects
/// from the dataset.
impl<DM: DesignMatrix, T: Targets> DatasetBase<DM, T> {
    /// Create a new dataset from design matrix and targets
    pub fn new(design_matrix: DM, targets: T) -> DatasetBase<DM, T> {
        DatasetBase {
            design_matrix,
            targets,
        }
    }

    /// Return references to targets
    pub fn targets(&self) -> &T {
        &self.targets
    }

    /// Return references to design matrix
    pub fn design_matrix(&self) -> &DM {
        &self.design_matrix
    }
}
