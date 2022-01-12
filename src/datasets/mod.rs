extern crate ndarray;

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
