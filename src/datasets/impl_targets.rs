use super::{DatasetBase, DesignMatrix, Targets};
use ndarray::{ArrayBase, Axis, Data, Dimension};

/// Implement the Targets trait for Ndarrays
impl<F, S, I> Targets for ArrayBase<S, I>
where
    S: Data<Elem = F>,
    I: Dimension,
{
    type Elem = F;

    fn n_tasks(&self) -> usize {
        self.len_of(Axis(1))
    }
}

/// Implement the Targets trait for DatasetBase
impl<F, DM, T> Targets for DatasetBase<DM, T>
where
    DM: DesignMatrix<Elem = F>,
    T: Targets<Elem = F>,
{
    type Elem = F;

    fn n_tasks(&self) -> usize {
        self.targets.n_tasks()
    }
}

/// Implement the Targets trait for an empty dataset
impl Targets for () {
    type Elem = ();

    fn n_tasks(&self) -> usize {
        0
    }
}

/// Implement the Targets trait for a reference
impl<T: Targets> Targets for &T {
    type Elem = T::Elem;

    fn n_tasks(&self) -> usize {
        (*self).n_tasks()
    }
}
