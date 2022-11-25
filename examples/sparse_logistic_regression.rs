extern crate sparseglm;

use sparseglm::{
    datafits::single_task::Logistic,
    datasets::DatasetBase,
    penalties::separable::MCP,
    solver::{CDSolver, Solver},
    utils::test_helpers::generate_random_data,
};

fn main() {
    let (x, y) = generate_random_data(30, 100);
    let dataset = DatasetBase::from((x, y));

    let alpha = 0.5;
    let gamma = 3.;

    // Regularizing logistic regression with a non-convex MCP penalty
    let mut datafit = Logistic::new();
    let penalty = MCP::new(alpha, gamma);
    let solver = Solver::new();

    println!("#### Fitting sparse logistic regression (MCP penalty)");
    let _ = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
}
