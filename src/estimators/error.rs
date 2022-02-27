use thiserror::Error;

/// Simplified `Result` using [`LassoError`](crate::LassoError) as error type
pub type Result<T> = std::result::Result<T, LassoError>;

/// Error variants from hyperparameter construction or model estimation
#[derive(Debug, Clone, Error)]
pub enum LassoError {
    /// The input has not enough samples
    #[error("invalid alpha {0}")]
    InvalidRegularization(f32),
    #[error("invalid gamma {0}")]
    InvalidGamma(f32),
    #[error("invalid tolerance {0}")]
    InvalidTolerance(f32),
    #[error("invalid K {0}")]
    InvalidK(usize),
    #[error("invalid p0 {0}")]
    InvalidP0(usize),
}
