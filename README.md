# rust-sparseglm

![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/cargo.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/pytest.yml/badge.svg)
![build](https://github.com/PABannier/rust-sparseglm/actions/workflows/build_doc.yml/badge.svg)

A fast and modular coordinate descent solver for sparse generalized linear models with **convex** and **non-convex**
penalties, written in Rust with Python bindings.

The algorithm is explained in depth in [CITE PAPER] and theoretical guarantees of convergence are given.
It also extensively demonstrates the superiority of this solver over existing ones.
A similar package written in pure Python can be found here: [CITE FLASHCD].

`rust-sparseglm` leverages [Anderson acceleration](https://github.com/mathurinm/andersoncd) and [working sets](https://github.com/mathurinm/celer) to propose a **fast** and **memory-efficient** solver on a large variety of algorithms. It can solve problems with millions of samples and features in seconds. It supports **dense** and **sparse** matrices (CSC arrays).

The philosophy of `rust-sparseglm` consists in offering a highly flexible API. Any sparse GLM can be implemented in under 50 lines of code by providing its datafit term and its penalty term, which makes it very easy to support new estimators.

```
// This is an example
```

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

We provide below a demonstration of `rust-sparseglm` against other fast coordinate descent solvers using the optimization benchmarking tool [Benchopt](https://github.com/benchopt/benchopt).

[INSERT IMAGE]
