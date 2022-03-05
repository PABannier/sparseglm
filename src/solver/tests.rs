use crate::datafits::*;
use crate::penalties::*;

use super::*;
use ndarray::array;

#[test]
fn test_lasso() {
    let x = array![[3., 2., 1.], [0.5, 3.4, 1.2]];
    let y = array![1.2, 4.5];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = Quadratic::new();
    let penalty = L1::new(0.1);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
    assert_eq!(
        coefficients,
        array![-0.4798204158784847, 1.3621219281657122, 0.]
    );
}

#[test]
fn test_mcp() {
    let x = array![[1.3, 2.1, 1.7], [-0.5, 1.4, 4.2]];
    let y = array![12., 42.];
    let dataset = DatasetBase::from((x, y));

    let mut datafit = Quadratic::new();
    let penalty = MCP::new(0.7, 2.);
    let solver = Solver::default();

    let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
    assert_eq!(coefficients, array![-17.81234024, 14.1919048, 3.14884837]);
}
