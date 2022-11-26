extern crate sparseglm;

use sparseglm::{
    datafits::single_task::Quadratic,
    datasets::DatasetBase,
    estimators::{estimators::Lasso, traits::Fit},
    penalties::separable::L1,
    solvers::{CDSolver, Solver},
    utils::{
        helpers::compute_alpha_max,
        test_helpers::{assert_array_all_close, generate_random_data},
    },
};

fn main() {
    let (x, y) = generate_random_data(30, 100);
    let alpha_max = compute_alpha_max(x.view(), y.view());
    let dataset = DatasetBase::from((x, y));

    let alpha = alpha_max * 0.1;

    // Penalty - Datafit - Solver API
    let mut datafit = Quadratic::new();
    let penalty = L1::new(alpha);
    let solver = Solver::new();

    println!("#### Fitting with Penalty - Datafit - Solver API...");
    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();

    // Estimator API
    println!("#### Fitting with the Estimator API...");
    let estimator = Lasso::params().alpha(alpha).fit(&dataset).unwrap();

    assert_array_all_close(coefficients.view(), estimator.coefficients(), 1e-4);
}
