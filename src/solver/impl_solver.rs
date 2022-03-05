use ndarray::Array1;
use std::fmt::Error;

use super::{CDSolver, Solver};

use crate::cd::coordinate_descent;
use crate::datafits::Datafit;
use crate::datasets::AsSingleTargets;
use crate::datasets::{DatasetBase, DesignMatrix};
use crate::penalties::Penalty;
use crate::Float;

impl<F: Float> Default for Solver<F> {
    fn default() -> Self {
        Solver {
            p0: 10,
            max_iterations: 50,
            max_epochs: 1000,
            tolerance: F::cast(1e-8),
            K: 5,
            use_acceleration: true,
            verbose: true,
        }
    }
}

impl<F> Solver<F> {
    /// The starting working set size.
    /// Defaults to `10`.
    pub fn p0(mut self, p0: usize) -> Self {
        self.p0 = p0;
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
            self.p0,
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
