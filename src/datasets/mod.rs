extern crate ndarray;

use ndarray::{ArrayBase, ArrayView1, ArrayView2, Axis, Ix2, OwnedRepr, ViewRepr};

use self::csc_array::CSCArray;

use thiserror::Error;

pub mod csc_array;
mod impl_datasets;
mod impl_design_matrix;
mod impl_targets;

pub struct DatasetBase<DM: DesignMatrix, T: AsMultiTargets> {
    pub design_matrix: DM,
    pub targets: T,
}

pub type DenseDataset<D, T> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, T>;

pub type DenseDatasetView<D, T> = DatasetBase<ArrayBase<ViewRepr<D>, Ix2>, T>;

pub type SparseDataset<'a, F, T> = DatasetBase<CSCArray<'a, F>, T>;

pub trait DesignMatrix: Sized {
    type Elem;

    fn n_features(&self) -> usize;

    fn matrix_type(&self) -> DesignMatrixType;
}

pub trait AsMultiTargets: Sized {
    type Elem;

    fn n_samples(&self) -> usize;

    fn n_tasks(&self) -> usize;

    fn as_multi_tasks(&self) -> ArrayView2<Self::Elem>;
}

pub trait AsSingleTargets: AsMultiTargets {
    fn try_single_target(&self) -> Result<ArrayView1<Self::Elem>, Error> {
        let multi_targets = self.as_multi_tasks();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::NonCastableSingleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}

pub enum DesignMatrixType {
    Dense,
    Sparse,
}

#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Unable to cast multi-dimensional array into 1d array")]
    NonCastableSingleTargets,
}
