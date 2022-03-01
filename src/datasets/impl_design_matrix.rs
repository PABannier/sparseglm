use super::{csc_array::CSCArray, DesignMatrix, DesignMatrixType};
use ndarray::{ArrayBase, Axis, Data, Ix2};

/// Implement DesignMatrix trait for NdArrays
impl<F, S: Data<Elem = F>> DesignMatrix for ArrayBase<S, Ix2> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.len_of(Axis(1))
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Dense
    }
}

/// Implement DesignMatrix trait for CSCArrays
impl<F> DesignMatrix for CSCArray<'_, F> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.indptr.len() - 1
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Sparse
    }
}

/// Implement records for references
impl<DM: DesignMatrix> DesignMatrix for &DM {
    type Elem = DM::Elem;

    fn n_features(&self) -> usize {
        (*self).n_features()
    }

    fn matrix_type(&self) -> DesignMatrixType {
        (*self).matrix_type()
    }
}
