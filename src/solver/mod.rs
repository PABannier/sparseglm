use std::fmt::Error;

use ndarray::Array1;

use super::Float;
use crate::datafits::Datafit;
use crate::datasets::AsSingleTargets;
use crate::datasets::{DatasetBase, DesignMatrix};
use crate::penalties::Penalty;

pub mod impl_solver;

#[derive(Debug, PartialEq, Clone)]
pub struct Solver<F> {
    pub p0: usize,
    pub max_iterations: usize,
    pub max_epochs: usize,
    pub tolerance: F,
    pub K: usize,
    pub use_acceleration: bool,
    pub verbose: bool,
}

pub trait CDSolver<F, DM, T, DF, P>
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
    ) -> Result<Array1<F>, Error>;
}
