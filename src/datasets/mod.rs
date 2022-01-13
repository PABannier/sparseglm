use ndarray::ViewRepr;

extern crate ndarray;

use ndarray::{ArrayBase, Ix2, OwnedRepr};

use self::csc_array::CSCArray;

pub mod csc_array;
mod impl_datasets;
mod impl_design_matrix;
mod impl_targets;

pub struct DatasetBase<DM, T>
where
    DM: DesignMatrix,
    T: Targets,
{
    pub design_matrix: DM,
    pub targets: T,
}

pub type DenseDataset<D, I> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, ArrayBase<OwnedRepr<D>, I>>;

pub type DenseDatasetView<'a, D, I> =
    DatasetBase<ArrayBase<ViewRepr<D>, Ix2>, ArrayBase<ViewRepr<D>, I>>;

pub type SparseDataset<'a, D, I> = DatasetBase<CSCArray<'a, D>, ArrayBase<OwnedRepr<D>, I>>;

pub type SparseDatasetView<'a, D, I> = DatasetBase<CSCArray<'a, D>, ArrayBase<ViewRepr<D>, I>>;

pub trait DesignMatrix: Sized {
    type Elem;

    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;

    fn matrix_type(&self) -> DesignMatrixType;
}

pub trait Targets: Sized {
    type Elem;

    fn n_tasks(&self) -> usize;
}

pub enum DesignMatrixType {
    Dense,
    Sparse,
}
