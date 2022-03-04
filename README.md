# rust-sparseglm

![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/cargo.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/pytest.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/build_doc.yml/badge.svg)

A fast and modular coordinate descent solver for sparse generalized linear models with **convex** and **non-convex**
penalties, written in Rust with Python bindings.

The algorithm is explained in depth in [CITE PAPER] and theoretical guarantees of convergence are given.
It also extensively demonstrates the superiority of this solver over existing ones.
A similar package written in pure Python can be found here: [FlashCD](https://github.com/mathurinm/flashcd).

`rust-sparseglm` leverages [Anderson acceleration](https://github.com/mathurinm/andersoncd) and [working sets](https://github.com/mathurinm/celer) to propose a **fast** and **memory-efficient** solver on a large variety of algorithms. It can solve problems with millions of samples and features in seconds. It supports **dense** and **sparse** matrices (CSC arrays).

The philosophy of `rust-sparseglm` consists in offering a highly flexible API. Any sparse GLM can be implemented in under 50 lines of code by providing its datafit term and its penalty term, which makes it very easy to support new estimators.

```rust
// Load some data and wrap them in a Dataset
let dataset = DatasetBase::from((X, y));

// Define a datafit (here a quadratic datafit for regression)
let mut datafit = Quadratic::default();

// Define a penalty (here a L1 + L2 penalty for ElasticNet)
let penalty = L1PlusL2::new(2., 0.3);

// Instantiate a Solver with default parameters
let solver = Solver::default();

// Solve the problem using coordinate descent
let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
```

For widely-known models like ElasticNet, `rust-sparseglm` already implements
those models and uses an API à la `Scikit-Learn`.

```rust
// Load some data and wrap them in a Dataset
let dataset = DatasetBase::from((X, y));

// Instantiate and fit the estimator
let estimator = ElasticNet::params()
                    .alpha(2.)
                    .l1_ratio(0.3)
                    .fit(&dataset)
                    .unwrap();

// Get the fitted coefficients
let coefficients = estimator.coefficients();
```

## Roadmap

Currently we support:

| Model                      |    Single task     |     Multi task     | Convexity  |
| -------------------------- | :----------------: | :----------------: | :--------: |
| Lasso                      | :heavy_check_mark: | :heavy_check_mark: |   Convex   |
| MCP                        | :heavy_check_mark: | :heavy_check_mark: | Non-convex |
| Elastic-Net                | :heavy_check_mark: | :heavy_check_mark: |   Convex   |
| L0.5                       | :heavy_check_mark: | :heavy_check_mark: | Non-convex |
| L2/3                       |         -          |         -          | Non-convex |
| SCAD                       |         -          |         -          | Non-convex |
| Indicator box              |         -          |         -          |   Convex   |
| Sparse logistic regression |         -          |         -          |   Convex   |
| Dual SVM with hinge loss   |         -          |         -          |   Convex   |

## Performance

We provide below a demonstration of `rust-sparseglm` against other fast coordinate descent solvers using the optimization benchmarking tool [Benchopt](https://github.com/benchopt/benchopt).

[INSERT IMAGE]
