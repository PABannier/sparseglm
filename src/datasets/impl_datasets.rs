use super::{csc_array::CSCArray, AsMultiTargets, DatasetBase, DesignMatrix};

use ndarray::{ArrayBase, Data, Ix2};

/// This implementation block provides a method for the creation of datasets
/// from dense matrices.
impl<F, D: Data<Elem = F>, T: AsMultiTargets> From<(ArrayBase<D, Ix2>, T)>
    for DatasetBase<ArrayBase<D, Ix2>, T>
{
    fn from(data: (ArrayBase<D, Ix2>, T)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}

/// This implementation block provides a method for the creation of datasets
/// from sparse matrices.
impl<'a, F, T: AsMultiTargets> From<(CSCArray<'a, F>, T)> for DatasetBase<CSCArray<'a, F>, T> {
    fn from(data: (CSCArray<'a, F>, T)) -> Self {
        DatasetBase {
            design_matrix: data.0,
            targets: data.1,
        }
    }
}

/// This implementation block provides methods to get record and target objects
/// from the dataset.
impl<DM: DesignMatrix, T: AsMultiTargets> DatasetBase<DM, T> {
    /// This method instantiates a new dataset from a design matrix and targets.
    pub fn new(design_matrix: DM, targets: T) -> DatasetBase<DM, T> {
        DatasetBase {
            design_matrix,
            targets,
        }
    }

    /// This method is a getter for the targets.
    pub fn targets(&self) -> &T {
        &self.targets
    }

    /// This method is a getter for the design matrix.
    pub fn design_matrix(&self) -> &DM {
        &self.design_matrix
    }
}
