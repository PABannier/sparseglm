# sparseglm

`sparseglm` is a Python wrapper for the Rust `sparseglm` crate.

This package is a coordinate descent solver to solve sparse generalized linear
models problems.

### Features

- Classification and regression datafits to support a wide-variety of models
- Convex and non-convex penalties (L1, MCP, L05, ...)
- Single-task and multi-task models supported
- A fast working set strategy coupled with non-linear extrapolation for faster
  convergence

### Installation

`sparseglm` requires Python 3.6+, `numpy` 1.15+ and can be installed with,

```
pip install sparseglm
```

## License

`sparseglm` is released under the [MIT License](https://github.com/PABannier/sparseglm/blob/main/LICENSE).
