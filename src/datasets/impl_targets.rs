use super::{AsMultiTargets, AsSingleTargets};
use ndarray::{ArrayBase, ArrayView2, Axis, Data, Ix1, Ix2};

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

impl<'a, F, S: Data<Elem = F>> AsSingleTargets for ArrayBase<S, Ix1> {}

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
