use super::{AsMultiTargets, AsSingleTargets};
use ndarray::{ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2};

/// This implements the [`AsMultiTargets`] trait for 2-dimensional targets.
impl<'a, F, S: Data<Elem = F>> AsMultiTargets for ArrayBase<S, Ix2> {
    type Elem = F;

    fn n_samples(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn n_tasks(&self) -> usize {
        self.len_of(Axis(1))
    }

    fn as_multi_tasks(&self) -> ArrayView2<F> {
        self.view()
    }
}

/// This implements the [`AsMultiTargets`] trait for 1-dimensional target.
impl<'a, F, S: Data<Elem = F>> AsMultiTargets for ArrayBase<S, Ix1> {
    type Elem = F;

    fn n_samples(&self) -> usize {
        self.len_of(Axis(0))
    }

    fn n_tasks(&self) -> usize {
        1
    }

    fn as_multi_tasks(&self) -> ArrayView2<F> {
        self.view().insert_axis(Axis(1))
    }
}

/// This implements the [`AsSingleTargets`] trait for 1-dimension target.
impl<'a, F, S: Data<Elem = F>> AsSingleTargets for ArrayBase<S, Ix1> {}

/// This implements the [`AsMultiTargets`] trait for references.
impl<T: AsMultiTargets> AsMultiTargets for &T {
    type Elem = T::Elem;

    fn n_samples(&self) -> usize {
        (*self).n_samples()
    }

    fn n_tasks(&self) -> usize {
        (*self).n_tasks()
    }

    fn as_multi_tasks(&self) -> ArrayView2<Self::Elem> {
        (*self).as_multi_tasks()
    }
}
