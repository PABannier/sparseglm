use thiserror::Error;

/// A shortcut type to return an [`EstimatorError`] when fitting an estimator.
pub type Result<T> = std::result::Result<T, EstimatorError>;

/// Error variants from hyperparameter construction or model estimation
#[derive(Debug, Clone, Error)]
pub enum EstimatorError {
    #[error("invalid alpha {0}")]
    InvalidRegularization(f32),
    #[error("invalid gamma {0}")]
    InvalidGamma(f32),
    #[error("invalid tolerance {0}")]
    InvalidTolerance(f32),
    #[error("invalid K {0}")]
    InvalidK(usize),
    #[error("invalid working set size {0}")]
    InvalidWSSize(usize),
    #[error("invalid l1 ratio {0}")]
    InvalidL1Ratio(f32),
}
