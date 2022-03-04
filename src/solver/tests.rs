use crate::datafits::*;
use crate::penalties::*;

use super::*;
use ndarray::array;

#[test]
fn test_lasso() {
    let x = array![[3., 2., 1.], [0.5, 3.4, 1.2]];
    let y = array![1.2, 4.5];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = Quadratic::default();
    let penalty = L1::new(0.7);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
    assert_eq!(coefficients, array![-0.15004725, 1.12181002, 0.]);
}

#[test]
fn test_mcp() {
    let x = array![[1.3, 2.1, 1.7], [-0.5, 1.4, 4.2]];
    let y = array![12., 42.];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = Quadratic::default();
    let penalty = MCP::new(0.7, 2.);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
    assert_eq!(coefficients, array![-2.60231414, 0., 9.53172209]);
}
