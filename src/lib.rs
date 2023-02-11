//! Fast and modular coordinate descent solver for sparse generalized linear models
//!
//! [`sparseglm`]'s philosophy consists in offering a highly flexible API. Any
//! sparse GLM can be implemented in under 50 lines of code by providing its datafit
//! term and its penalty term, which makes it very easy to support new estimators.
//!
//! A quick example on how to solve a sparse GLM optimization problem by choosing an
//! arbitrary combination of [`Datafit`] and [`Penalty`].
//!
//! ```ignore
//! // Load some data and wrap them in a Dataset
//! let dataset = DatasetBase::from((x, y));
//!
//! // Define a datafit (here a quadratic datafit for regression)
//! let mut datafit = Quadratic::new();
//!
//! // Define a penalty (here a L1 penalty for Lasso)
//! let penalty = L1::new(0.7);
//!
//! // Instantiate a Solver with default parameters
//! let solver = Solver::new();
//!
//! // Solve the problem using coordinate descent
//! let coefficients = solver.solve(dataset, &mut datafit, &penalty).unwrap();
//! ```
//!
//! For widely-used models like Lasso, [`sparseglm`] already implements
//! those models.
//!
//! ```ignore
//! // Load some data and wrap them in a Dataset
//! let dataset = DatasetBase::from((x, y));
//!
//! // Instantiate and fit the estimator
//! let estimator = Lasso::params()
//!                   .alpha(0.7)
//!                   .fit(&dataset)
//!                   .unwrap();
//!
//! // Get the fitted coefficients
//! let coefficients = estimator.coefficients();
//!

#![allow(non_snake_case)]

extern crate approx;
extern crate ndarray;
extern crate ndarray_stats;
extern crate num_traits;
extern crate rand;
extern crate rand_distr;
extern crate sprs;
extern crate thiserror;

use ndarray::ScalarOperand;

use num_traits::{AsPrimitive, FromPrimitive, NumAssignOps, NumCast, Signed};

use std::cmp::PartialOrd;
use std::fmt;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Float point numbers
///
/// This trait bound multiplexes to the most common assumption of floating point
/// number and implement them for 32bit and 64bit float points.
/// Ref: `<https://github.com/rust-ml/linfa/blob/master/src/dataset/mod.rs#L36>`
pub trait Float:
    FromPrimitive
    + num_traits::Float
    + PartialOrd
    + Sync
    + Send
    + Default
    + fmt::Display
    + fmt::Debug
    + Signed
    + Sum
    + NumAssignOps
    + AsPrimitive<usize>
    + for<'a> AddAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + num_traits::MulAdd<Output = Self>
    + ScalarOperand
    + approx::AbsDiffEq
{
    #[cfg(feature = "ndarray-linalg")]
    type Lapack: Float + Scalar + Lapack;
    #[cfg(not(feature = "ndarray-linalg"))]
    type Lapack: Float;

    fn cast<T: NumCast>(x: T) -> Self {
        NumCast::from(x).unwrap()
    }
}

impl Float for f32 {
    type Lapack = f32;
}

impl Float for f64 {
    type Lapack = f64;
}

pub mod datafits;
pub mod datasets;
pub mod estimators;
pub mod penalties;
pub mod solvers;
pub mod utils;
