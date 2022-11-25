extern crate sparseglm;

use sparseglm::{
    datafits::single_task::Quadratic,
    datasets::DatasetBase,
    estimators::{estimators::ElasticNet, traits::Fit},
    penalties::separable::L1PlusL2,
    solvers::{CDSolver, Solver},
    utils::test_helpers::{assert_array_all_close, generate_random_data},
};

fn main() {
    let (x, y) = generate_random_data(100, 60);
    let dataset = DatasetBase::from((x, y));

    let alpha = 0.05;
    let l1_ratio = 0.7;

    // Penalty - Datafit - Solver API
    let mut datafit = Quadratic::new();
    let penalty = L1PlusL2::new(alpha, l1_ratio);
    let solver = Solver::new();

    println!("#### Fitting with Penalty - Datafit - Solver API...");
    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();

    // Estimator API
    println!("#### Fitting with the Estimator API...");
    let estimator = ElasticNet::params()
        .alpha(alpha)
        .l1_ratio(l1_ratio)
        .fit(&dataset)
        .unwrap();

    assert_array_all_close(coefficients.view(), estimator.coefficients(), 1e-4);
}
