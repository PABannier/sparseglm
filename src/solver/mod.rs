use std::fmt::Error;

use ndarray::{Array1, Array2};

use super::Float;
use crate::datafits::multi_task::MultiTaskDatafit;
use crate::datafits::single_task::Datafit;
use crate::datasets::{AsMultiTargets, AsSingleTargets};
use crate::datasets::{DatasetBase, DesignMatrix};
use crate::penalties::Penalty;
use crate::penalties_multitask::MultiTaskPenalty;

#[cfg(test)]
pub mod tests;

pub mod impl_solver;

/// [`sparseglm`] offers two ways to solve optimization problems. Either
/// using a pre-defined estimator and using an API à la Scikit-Learn, or using
/// a [`Solver`] jointly with a [`Datafit`] and a [`Penalty`] object. The
/// [`Solver`] object contains all the hyperparameters needed for the optimization
/// routine.
#[derive(Debug, PartialEq, Clone)]
pub struct Solver<F> {
    /// The start size of the working set
    pub p0: usize,
    /// The maximum number of iterations in the outer loop
    pub max_iterations: usize,
    /// The maximum number of epochs in the inner loop
    pub max_epochs: usize,
    /// The tolerance for the suboptimality gap
    pub tolerance: F,
    /// The number of iterates used to construct an extrapolated point
    pub K: usize,
    /// The use of Anderson acceleration
    pub use_acceleration: bool,
    /// The verbosity of the solver
    pub verbose: bool,
}

/// This trait calls the [`coordinate_descent`] backbone function to solve an
/// optimization problem given a [`Penalty`], a [`Datafit`] and a [`Solver`].
/// They contain all the hyperparameters needed by the coordinate descent
/// solver function.
pub trait CDSolver<F, DM, T, DF, P>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    /// This method calls the [`coordinate_descent`] backbone function.
    fn solve(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &mut DF,
        penalty: &P,
    ) -> Result<Array1<F>, Error>;
}

pub trait BCDSolver<F, DM, T, DF, P>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: MultiTaskPenalty<F>,
{
    /// This method calls the [`block_coordinate_descent`] backbone function.
    fn solve_multi_task(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &mut DF,
        penalty: &P,
    ) -> Result<Array2<F>, Error>;
}
