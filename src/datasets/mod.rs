use ndarray::{ArrayBase, ArrayView1, ArrayView2, Axis, Ix2, OwnedRepr, ViewRepr};
use thiserror::Error;

use self::csc_array::CSCArray;

pub mod csc_array;
mod impl_datasets;
mod impl_design_matrix;
mod impl_targets;

/// [`DatasetBase`] is a basic structure that holds a design matrix and a target
/// vector or matrix. It is the most basic building block needed to use the
/// estimators offered by [`rust-sparseglm`].
///
/// It is inspired from: `https://github.com/rust-ml/linfa`.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetBase<DM: DesignMatrix, T: AsMultiTargets> {
    pub design_matrix: DM,
    pub targets: T,
}

/// Design matrices can either be dense matrices whose data is stored in standard
/// arrays, or sparse matrices stored in [`CSCArray`] to leverage their very
/// sparse structure.
#[derive(Debug, Clone, PartialEq)]
pub enum DesignMatrixType {
    Dense,
    Sparse,
}

/// Conversion error when trying to cast a 2-dimensional matrix into a
/// 1-dimensional vector.
#[derive(Error, Debug, Clone)]
pub enum Error {
    #[error("Unable to cast multi-dimensional array into 1d array")]
    NonCastableSingleTargets,
}

/// A Dataset with an owned dense design matrix.
pub type DenseDataset<D, T> = DatasetBase<ArrayBase<OwnedRepr<D>, Ix2>, T>;

/// A Dataset with a dense design matrix view.
pub type DenseDatasetView<D, T> = DatasetBase<ArrayBase<ViewRepr<D>, Ix2>, T>;

/// A Dataset with an owned sparse design matrix.
pub type SparseDataset<'a, F, T> = DatasetBase<CSCArray<'a, F>, T>;

/// This trait implements the basic methods for an array to be considered a
/// design matrix.
pub trait DesignMatrix: Sized {
    /// This indicates the element type held in the design matrix (usually a float).
    type Elem;

    /// This gives the number of features in the design matrix.
    fn n_features(&self) -> usize;

    /// This gives the matrix type (either dense or sparse) of the design matrix.
    fn matrix_type(&self) -> DesignMatrixType;
}

/// This trait implements the basic methods for an array to be considered
/// a target matrix.
pub trait AsMultiTargets: Sized {
    /// This indicates the element type held in the target matrix.
    type Elem;

    /// This gives the number of samples in the target array.
    fn n_samples(&self) -> usize;

    /// This gives the number of tasks in the target array.
    fn n_tasks(&self) -> usize;

    /// This method allows to cast a 1-dimensional target vector into a 2-dimensional
    /// target matrix.
    fn as_multi_tasks(&self) -> ArrayView2<Self::Elem>;
}

/// This trait is a sub-trait of [`AsMultiTargets`] and is specifically designed
/// to hold target vectors, not target matrices. It is used by single-task estimators
/// like [`Lasso`] or [`MCPEstimator`].
pub trait AsSingleTargets: AsMultiTargets {
    /// This method tries to cast an array into a 1-dimensional array (a target vector).
    /// Matrices with more than 1 columns can't be casted to vectors. A
    /// [`Error::NonCastableSingleTargets`] is thrown.
    fn try_single_target(&self) -> Result<ArrayView1<Self::Elem>, Error> {
        let multi_targets = self.as_multi_tasks();

        if multi_targets.len_of(Axis(1)) > 1 {
            return Err(Error::NonCastableSingleTargets);
        }

        Ok(multi_targets.index_axis_move(Axis(1), 0))
    }
}
