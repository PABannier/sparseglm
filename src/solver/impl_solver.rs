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
    pub fn p0(mut self, p0: usize) -> Self {
        self.p0 = p0;
        self
    }

    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn K(mut self, K: usize) -> Self {
        self.K = K;
        self
    }

    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.use_acceleration = use_acceleration;
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

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
