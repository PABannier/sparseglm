#![allow(non_snake_case)]

#![feature(test)]
extern crate test;

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
/// Ref: https://github.com/rust-ml/linfa/blob/master/src/dataset/mod.rs#L36
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

pub mod bcd;
pub mod cd;
pub mod datafits;
pub mod datafits_multitask;
pub mod datasets;
pub mod estimators;
pub mod helpers;
pub mod penalties;
pub mod penalties_multitask;
pub mod solver;
pub mod solver_multitask;
