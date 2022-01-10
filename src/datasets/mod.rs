extern crate ndarray;

mod csc_array;
mod impl_datasets;
mod impl_design_matrix;

pub struct DatasetBase<DM, T>
where
    DM: DesignMatrix,
{
    pub design_matrix: DM,
    pub targets: T,
}

pub trait DesignMatrix: Sized {
    type Elem;

    fn n_samples(&self) -> usize;
    fn n_features(&self) -> usize;
}

pub trait Targets: Sized {
    type Elem;

    fn n_samples(&self) -> usize;
    fn n_tasks(&self) -> usize;
}
