extern crate rust_sparseglm;

use rust_sparseglm::{
    datafits::Quadratic,
    datasets::DatasetBase,
    estimators::{estimators::Lasso, traits::Fit},
    helpers::test_helpers::generate_random_data,
    penalties::L1,
    solver::{CDSolver, Solver},
};

fn main() {
    let (x, y) = generate_random_data(30, 100);
    let dataset = DatasetBase::from((x, y));

    // Penalty - Datafit - Solver API
    let mut datafit = Quadratic::new();
    let penalty = L1::new(1.);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();

    // Estimator API
    let estimator = Lasso::params().alpha(1.).fit(&dataset).unwrap();

    // Compare coefficients
    assert_eq!(coefficients, estimator.coefficients());
}
