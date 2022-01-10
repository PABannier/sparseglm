use super::{csc_array::CSCArray, DatasetBase, DesignMatrix, Targets};
use ndarray::{ArrayBase, Axis, Data, Dimension};

/// Implement DesignMatrix trait for NdArrays
impl<F, S: Data<Elem = F>, I: Dimension> DesignMatrix for ArrayBase<S, I> {
    type Elem = F;

    fn n_samples(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn n_features(&self) -> usize {
        self.len_of(Axis(1))
    }
}

/// Implement DesignMatrix trait for CSCArrays
impl<F> DesignMatrix for CSCArray<'_, F> {
    type Elem = F;

    fn n_samples(&self) -> usize {
        0 // TODO: Check????
    }

    fn n_features(&self) -> usize {
        self.indptr.len() - 1
    }
}

/// Implement DesignMatrix trait for DatasetBase
impl<F, DM, T> DesignMatrix for DatasetBase<DM, T>
where
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    type Elem = F;

    fn n_samples(&self) -> usize {
        self.design_matrix.n_samples()
    }

    fn n_features(&self) -> usize {
        self.design_matrix.n_features()
    }
}

/// Implement DesignMatrix for an empty dataset
impl DesignMatrix for () {
    type Elem = ();

    fn n_samples(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        0
    }
}

/// Implement records for references
impl<DM: DesignMatrix> DesignMatrix for &DM {
    type Elem = DM::Elem;

    fn n_samples(&self) -> usize {
        (*self).n_samples()
    }

    fn n_features(&self) -> usize {
        (*self).n_features()
    }
}
