# rust-sparseglm

![build](https://github.com/PABannier/RustyLasso/actions/workflows/cargo.yml/badge.svg)

A fast and modular coordinate descent solver for sparse generalized linear models with **convex** and **non-convex** separable penalties, written in Rust with Python bindings.

The solver implemented in this crate is explained in depth in [CITE PAPER] and provides theoretical guarantees on the convergence of such solver.
It also extensively demonstrates the superiority of such solver over non-accelerated approaches. A similar package written in pure Python
(with Numba JIT compiled code) can be found here: [CITE FLASHCD].

This solver leverages ![**Anderson acceleration**](https://github.com/mathurinm/andersoncd) and ![**working sets**](https://github.com/mathurinm/celer)
to achieve state-of-the-art performance on a large variety of algorithms. It can solve problems with millions of samples and features in seconds.

The crate is organized in such a way it is very modular. Any sparse GLM can be implemented in under 50 lines of code by providing **a datafit** and **a penalty**, which makes it very easy to support new estimators.

Currently we support:

| Model                      |    Single task     |     Multi task     | Convexity  |
| -------------------------- | :----------------: | :----------------: | :--------: |
| Lasso                      | :heavy_check_mark: | :heavy_check_mark: |   Convex   |
| MCP                        | :heavy_check_mark: | :heavy_check_mark: | Non-convex |
| Elastic-Net                |         -          |         -          |   Convex   |
| L0.5                       | :heavy_check_mark: | :heavy_check_mark: | Non-convex |
| L2/3                       |         -          |         -          | Non-convex |
| SCAD                       |         -          |         -          | Non-convex |
| Indicator box              |         -          |         -          |   Convex   |
| Sparse logistic regression |         -          |         -          |   Convex   |
| Dual SVM with hinge loss   |         -          |         -          |   Convex   |

We provide below a demonstration of rust-sparseglm against other fast coordinate descent solvers using the optimization benchmarking tool
![Benchopt](https://github.com/benchopt/benchopt).

[INSERT IMAGE]

## Quick demo

ADD SNIPPET OF CODE TO FIT

## Crate architecture

rust-sparseglm is organized around two backbone functions `coordinate_descent` and `block_coordinate_descent`. Among others, both functions
receive 3 structs which are responsible for the optimization routine: a `Solver`, a `Datafit` and a `Penalty`.

`Datafit` pre-computes quantities that will be used during the descent phase. `Penalty` implements the `Penalty` trait and computes the next
proximal step and computes the distance of the gradient to the subdifferential, a crucial score to rank the features that should be included in the
working set.

**\*Why not use duality theory to build working sets and use the duality gap as a stopping criterion?**

Convex models offer stronger theoretical guarantees and tools to write memory-efficient and fast optimization routine. However, building a solver
around duality theory restrains de facto the range of supported models to only convex ones. The philosohpy of rust-sparseglm is to support as many models
as possible in the fewest lines of code. Therefore, we can't rely on duality theory.

Instead of duality theory, rust-sparseglm leverages two criteria to build working sets and stop the optimization routine: - the distance of the gradient of the datafit to the subdifferential of the penalty - the violation of the fixed-point equation

The first one is used by most models and is implemented by the `subdiff_distance` function in the `Penalty` trait. The second one is used when the first
strategy fails. For instance, the subdifferential of the L0.5-quasinorm penalty at 0 is the real-line, which makes the computation of the distance of the
gradient of the datafit with respect to feature `Xj` useless since the distance is always 0. Therefore, the first criterion becomes uninformative and we need
to rely on the second criterion.
