use ndarray::{Array1, Array2};
use std::fmt::Error;

use super::{BCDSolver, CDSolver, Solver};

use crate::datafits::multi_task::MultiTaskDatafit;
use crate::datafits::single_task::Datafit;
use crate::datasets::{AsMultiTargets, AsSingleTargets, DatasetBase, DesignMatrix};
use crate::penalties::block_separable::MultiTaskPenalty;
use crate::penalties::separable::Penalty;
use crate::solvers::anderson_bcd::block_coordinate_descent;
use crate::solvers::anderson_cd::coordinate_descent;
use crate::Float;

impl<F: Float> Default for Solver<F> {
    fn default() -> Self {
        Solver {
            ws_start_size: 10,
            max_iterations: 50,
            max_epochs: 1000,
            tolerance: F::cast(1e-8),
            K: 5,
            use_acceleration: true,
            verbose: true,
        }
    }
}

impl<F: Float> Solver<F> {
    /// Create a Solver instance with default parameters.
    pub fn new() -> Self {
        Default::default()
    }

    /// The starting working set size.
    /// Defaults to `10`.
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.ws_start_size = ws_start_size;
        self
    }

    /// The maximum number of iterations in the outer loop. The outer loop
    /// is responsible for growing the working set.
    /// Default to `50`.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// The maximum number of epochs in the inner loop. The inner loop
    /// is responsible for applying coordinate descent to a subproblem restricted
    /// to the working set. It is the number of iterations used during the
    /// descent phase.
    /// Defaults to `1000`.
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    /// The tolerance below which the optimization procedure is stopped.
    /// Defaults to `1e-8`.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// The number of past iterates used to construct an extrapolated point
    /// with Anderson acceleration.
    /// Defaults to `5`.
    pub fn K(mut self, K: usize) -> Self {
        self.K = K;
        self
    }

    /// The use of acceleration to accelerate the optimization procedure.
    /// Defaults to `true`.
    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.use_acceleration = use_acceleration;
        self
    }

    /// The verbosity level of the solver. If `true`, the solver prints the
    /// objective value as well as the working set during the fitting procedure.
    /// Defaults to `true`.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// This implements the [`CDSolver`] trait by calling the [`coordinate_descent`]
/// backbone function for single-task optimization problems.
impl<F, DM, T, DF, P> CDSolver<F, DM, T, DF, P> for Solver<F>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsSingleTargets<Elem = F>,
    DF: Datafit<F, DM, T>,
    P: Penalty<F>,
{
    fn solve(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &mut DF,
        penalty: &P,
    ) -> Result<Array1<F>, Error> {
        let w = coordinate_descent(
            dataset,
            datafit,
            penalty,
            self.ws_start_size,
            self.max_iterations,
            self.max_epochs,
            self.tolerance,
            self.K,
            self.use_acceleration,
            self.verbose,
        );

        Ok(w)
    }
}

/// This implements the [`BCDSolver`] trait by calling the [`block_coordinate_descent`]
/// backbone function for multi-task optimization problems.
impl<F, DM, T, DF, P> BCDSolver<F, DM, T, DF, P> for Solver<F>
where
    F: Float,
    DM: DesignMatrix<Elem = F>,
    T: AsMultiTargets<Elem = F>,
    DF: MultiTaskDatafit<F, DM, T>,
    P: MultiTaskPenalty<F>,
{
    fn solve_multi_task(
        &self,
        dataset: &DatasetBase<DM, T>,
        datafit: &mut DF,
        penalty: &P,
    ) -> Result<Array2<F>, Error> {
        let W = block_coordinate_descent(
            dataset,
            datafit,
            penalty,
            self.ws_start_size,
            self.max_iterations,
            self.max_epochs,
            self.tolerance,
            self.K,
            self.use_acceleration,
            self.verbose,
        );

        Ok(W)
    }
}
