use super::{csc_array::CSCArray, DatasetBase, DesignMatrix, DesignMatrixType, Targets};
use ndarray::{ArrayBase, Axis, Data, Dimension};

/// Implement DesignMatrix trait for NdArrays
impl<F, S: Data<Elem = F>, I: Dimension> DesignMatrix for ArrayBase<S, I> {
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

/// Implement DesignMatrix trait for DatasetBase
impl<F, DM, T> DesignMatrix for DatasetBase<DM, T>
where
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    type Elem = F;

    fn n_features(&self) -> usize {
        self.design_matrix.n_features()
    }

    fn matrix_type(&self) -> DesignMatrixType {
        self.design_matrix.matrix_type()
    }
}

/// Implement DesignMatrix for an empty dataset
impl DesignMatrix for () {
    type Elem = ();

    fn n_features(&self) -> usize {
        0
    }

    fn matrix_type(&self) -> DesignMatrixType {
        // By default, design matrices are assumed dense
        DesignMatrixType::Dense
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
