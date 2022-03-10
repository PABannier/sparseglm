# sparseglm

![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/cargo.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/pytest.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/build_doc.yml/badge.svg)

A fast and modular coordinate descent solver for sparse generalized linear models
with **convex** and **non-convex** penalties.

The details of the `rust-sparseglm` solver are explored in [CITE PAPER]. It provides
theoretical guarantees of convergence and extensively demonstrates the superiority
of this solver over existing ones. A similar package written in pure Python can be found
here: [FlashCD](https://github.com/mathurinm/flashcd).

`rust-sparseglm` leverages [Anderson acceleration](https://github.com/mathurinm/andersoncd)
and [working sets](https://github.com/mathurinm/celer) to propose a **fast** and
**memory-efficient** solver on a wide variety of algorithms. It can solve problems
with millions of samples and features in seconds. It supports **dense** and
**sparse** matrices via CSC arrays.

The philosophy of `rust-sparseglm` consists in offering a highly flexible API.
Any sparse GLM can be implemented in under 50 lines of code by providing its datafit
term and its penalty term, which makes it very easy to support new estimators.

```rust
// Load some data and wrap them in a Dataset
let dataset = DatasetBase::from((x, y));

// Define a datafit (here a quadratic datafit for regression)
let mut datafit = Quadratic::new();

// Define a penalty (here a L1 penalty for Lasso)
let penalty = L1::new(0.7);

// Instantiate a Solver with default parameters
let solver = Solver::default();

// Solve the problem using coordinate descent
let coefficients = solver.solve(&dataset, &mut datafit, &penalty).unwrap();
```

For widely-known models like Lasso, `rust-sparseglm` already implements
these estimators and offers an API à la `Scikit-Learn`.

```rust
// Load some data and wrap them in a Dataset
let dataset = DatasetBase::from((x, y));

// Instantiate and fit the estimator
let estimator = Lasso::params()
                  .alpha(2.)
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
| Sparse logistic regression | :heavy_check_mark: |         -          |   Convex   |
| Dual SVM with hinge loss   |         -          |         -          |   Convex   |

## Performance

We provide below a demonstration of `rust-sparseglm` against other fast coordinate
descent solvers using the optimization benchmarking tool [Benchopt](https://github.com/benchopt/benchopt).

[INSERT IMAGE]

## Building the Python package

This repo includes Python bindings to run the existing estimators (in the `Estimators`crate)
in a Python environment. To install it, run at the root of the repo:

```bash

# Install requirements
pip install -r requirements.txt

# Compile and build Python wheel
cd python
python ./setup.py bdist_wheel
```
