extern crate sparseglm;

use sparseglm::{
    datafits::Logistic,
    datasets::DatasetBase,
    helpers::test_helpers::generate_random_data,
    penalties::MCP,
    solver::{CDSolver, Solver},
};

fn main() {
    let (x, y) = generate_random_data(30, 100);
    let dataset = DatasetBase::from((x, y));

    let alpha = 0.5;
    let gamma = 3.;

    // Regularizing logistic regression with a non-convex MCP penalty
    let mut datafit = Logistic::new();
    let penalty = MCP::new(alpha, gamma);
    let solver = Solver::default();

    println!("#### Fitting sparse logistic regression (MCP penalty)");
    let _ = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
}
