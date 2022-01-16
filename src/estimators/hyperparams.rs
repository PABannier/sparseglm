use super::error::{LassoError, Result};
use super::param_guard::ParamGuard;
use super::Float;

/// A verified hyperparameter set ready for the fitting of a Lasso regression model
///

pub struct LassoValidParams<F> {
    alpha: F,
    max_iterations: usize,
    max_epochs: usize,
    p0: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> LassoValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn p0(&self) -> usize {
        self.p0
    }

    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    pub fn K(&self) -> usize {
        self.K
    }

    pub fn use_acceleration(&self) -> bool {
        self.use_acceleration
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }
}

/// A hyper-parameter set during construction
///
/// Configures and minimizes the following objective function:
/// ```ignore
/// 1 / (2 * n_samples) * ||y - Xw||^2_2
///     + alpha * ||w||_1
/// ```
///
pub struct LassoParams<F>(LassoValidParams<F>);

impl<F: Float> Default for LassoParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a Lasso model
impl<F: Float> LassoParams<F> {
    /// Create default Lasso hyper parameters
    pub fn new() -> LassoParams<F> {
        Self(LassoValidParams {
            alpha: F::one(),
            max_iterations: 50,
            max_epochs: 1000,
            p0: 10,
            tolerance: F::cast(1e-6),
            K: 5,
            use_acceleration: true,
            verbose: true,
        })
    }

    /// Set the regularization hyperparameter. A higher value yields sparser
    /// solutions.
    /// Defaults to `1` if not set.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations in the outer loop used to build
    /// working set.
    /// Defaults to `50` if not set.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.0.max_iterations = max_iterations;
        self
    }

    /// Set the maximum number of epochs in the inner loop during the descent
    /// routine.
    /// Defaults to `1000` if not set.
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.0.max_epochs = max_epochs;
        self
    }

    /// Set the initial working set size.
    ///
    /// Defaults to `10` if not set.
    pub fn p0(mut self, p0: usize) -> Self {
        self.0.p0 = p0;
        self
    }

    /// Set the stopping criterion for the optimization routine (KKT violation).
    ///
    /// Defaults to `1e-6` if not set.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Set the number of points used for extrapolation.
    ///
    /// Defaults to `5` if not set.
    pub fn K(mut self, K: usize) -> Self {
        self.0.K = K;
        self
    }

    /// Enables the use of Anderson acceleration for the extrapolation of
    /// primal iterates.
    /// Defaults to `true` if not set.
    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.0.use_acceleration = use_acceleration;
        self
    }

    /// Sets the verbosity level of the solver.
    ///
    /// Defaults to `true` if not set.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.verbose = verbose;
        self
    }
}

impl<F: Float> ParamGuard for LassoParams<F> {
    type Checked = LassoValidParams<F>;
    type Error = LassoError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(LassoError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(LassoError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(LassoError::InvalidK(self.0.K))
        } else if self.0.p0 <= 0 {
            Err(LassoError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// Configure a MultiTaskLasso model.
pub struct MultiTaskLassoValidParams<F> {
    alpha: F,
    max_iterations: usize,
    max_epochs: usize,
    p0: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> MultiTaskLassoValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn p0(&self) -> usize {
        self.p0
    }

    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    pub fn K(&self) -> usize {
        self.K
    }

    pub fn use_acceleration(&self) -> bool {
        self.use_acceleration
    }

    pub fn verbose(&self) -> bool {
        self.verbose
    }
}

/// A hyper-parameter set during construction
///
/// Configures and minimizes the following objective function:
/// ```ignore
/// 1 / (2 * n_samples) * ||Y - XW||^2_F
///     + alpha * sum_j ||w_j||_2
/// ```
///
pub struct MultiTaskLassoParams<F>(MultiTaskLassoValidParams<F>);

impl<F: Float> Default for MultiTaskLassoParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a Lasso model
impl<F: Float> MultiTaskLassoParams<F> {
    /// Create default Lasso hyper parameters
    pub fn new() -> MultiTaskLassoParams<F> {
        Self(MultiTaskLassoValidParams {
            alpha: F::one(),
            max_iterations: 50,
            max_epochs: 1000,
            p0: 10,
            tolerance: F::cast(1e-6),
            K: 5,
            use_acceleration: true,
            verbose: true,
        })
    }

    /// Set the regularization hyperparameter. A higher value yields sparser
    /// solutions.
    /// Defaults to `1` if not set.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Set the maximum number of iterations in the outer loop used to build
    /// working set.
    /// Defaults to `50` if not set.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.0.max_iterations = max_iterations;
        self
    }

    /// Set the maximum number of epochs in the inner loop during the descent
    /// routine.
    /// Defaults to `1000` if not set.
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.0.max_epochs = max_epochs;
        self
    }

    /// Set the initial working set size.
    ///
    /// Defaults to `10` if not set.
    pub fn p0(mut self, p0: usize) -> Self {
        self.0.p0 = p0;
        self
    }

    /// Set the stopping criterion for the optimization routine (KKT violation).
    ///
    /// Defaults to `1e-6` if not set.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.tolerance = tolerance;
        self
    }

    /// Set the number of points used for extrapolation.
    ///
    /// Defaults to `5` if not set.
    pub fn K(mut self, K: usize) -> Self {
        self.0.K = K;
        self
    }

    /// Enables the use of Anderson acceleration for the extrapolation of
    /// primal iterates.
    /// Defaults to `true` if not set.
    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.0.use_acceleration = use_acceleration;
        self
    }

    /// Sets the verbosity level of the solver.
    ///
    /// Defaults to `true` if not set.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.verbose = verbose;
        self
    }
}

impl<F: Float> ParamGuard for MultiTaskLassoParams<F> {
    type Checked = MultiTaskLassoValidParams<F>;
    type Error = LassoError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(LassoError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(LassoError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(LassoError::InvalidK(self.0.K))
        } else if self.0.p0 <= 0 {
            Err(LassoError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
