use super::{csc_array::CSCArray, DesignMatrix, DesignMatrixType};
use ndarray::{ArrayBase, Axis, Data, Ix2};

/// This implements the [`DesignMatrix`] trait for dense matrices.
impl<F, S: Data<Elem = F>> DesignMatrix for ArrayBase<S, Ix2> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.len_of(Axis(1))
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Dense
    }
}

/// This implements the [`DesignMatrix`] trait for sparse matrices.
impl<F> DesignMatrix for CSCArray<'_, F> {
    type Elem = F;

    fn n_features(&self) -> usize {
        self.indptr.len() - 1
    }

    fn matrix_type(&self) -> DesignMatrixType {
        DesignMatrixType::Sparse
    }
}

/// This implements the [`DesignMatrix`] trait for references.
impl<DM: DesignMatrix> DesignMatrix for &DM {
    type Elem = DM::Elem;

    fn n_features(&self) -> usize {
        (*self).n_features()
    }

    fn matrix_type(&self) -> DesignMatrixType {
        (*self).matrix_type()
    }
}
