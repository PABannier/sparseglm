extern crate sparseglm;

use sparseglm::{
    datafits::multi_task::QuadraticMultiTask,
    datasets::DatasetBase,
    estimators::{estimators::MultiTaskLasso, traits::Fit},
    penalties::block_separable::L21,
    solver::{BCDSolver, Solver},
    utils::{
        helpers::compute_alpha_max_mtl,
        test_helpers::{assert_array2d_all_close, generate_random_data_mtl},
    },
};

fn main() {
    let (x, y) = generate_random_data_mtl(30, 100, 40);
    let alpha_max = compute_alpha_max_mtl(x.view(), y.view());
    let dataset = DatasetBase::from((x, y));

    let alpha = alpha_max * 0.1;

    // Penalty - Datafit - Solver API
    let mut datafit = QuadraticMultiTask::new();
    let penalty = L21::new(alpha);
    let solver = Solver::new();

    println!("#### Fitting with Penalty - Datafit - Solver API...");
    let coefficients = solver
        .solve_multi_task(&dataset, &mut datafit, &penalty)
        .unwrap();

    // Estimator API
    println!("#### Fitting with the Estimator API...");
    let estimator = MultiTaskLasso::params().alpha(alpha).fit(&dataset).unwrap();

    assert_array2d_all_close(coefficients.view(), estimator.coefficients(), 1e-4);
}
