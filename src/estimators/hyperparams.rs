use super::error::{EstimatorError, Result};
use super::param_guard::ParamGuard;
use super::Float;

/// A verified hyperparameter set ready for fitting a sparse GLM

pub struct SolverParams<F> {
    max_iterations: usize,
    max_epochs: usize,
    p0: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

pub trait SolverValidParams {
    type Elem;

    fn max_iterations(&self) -> usize;

    fn max_epochs(&self) -> usize;

    fn p0(&self) -> usize;

    fn tolerance(&self) -> Self::Elem;

    fn K(&self) -> usize;

    fn use_acceleration(&self) -> bool;

    fn verbose(&self) -> bool;
}

impl<F> SolverValidParams for SolverParams<F> {
    type Elem = F;

    fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    fn p0(&self) -> usize {
        self.p0
    }

    fn tolerance(&self) -> Self::Elem {
        self.tolerance
    }

    fn K(&self) -> usize {
        self.K
    }

    fn use_acceleration(&self) -> bool {
        self.use_acceleration
    }

    fn verbose(&self) -> bool {
        self.verbose
    }
}

impl<F: Float> Default for SolverParams<F> {
    fn default() -> Self {
        SolverParams {
            max_iterations: 50,
            max_epochs: 1000,
            p0: 10,
            tolerance: F::cast(1e-6),
            K: 5,
            use_acceleration: true,
            verbose: true,
        }
    }
}

impl<F: Float> ParamGuard for SolverParams<F> {
    type Checked = SolverParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.tolerance.to_f32().unwrap(),
            ))
        } else if self.K <= 0 {
            Err(EstimatorError::InvalidK(self.K))
        } else if self.p0 <= 0 {
            Err(EstimatorError::InvalidP0(self.p0))
        } else {
            Ok(&self)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self)
    }
}

/// A verified hyperparameter set ready for the fitting of a Lasso regression model
///

pub struct LassoValidParams<F> {
    alpha: F,
    solver_params: SolverParams<F>,
}

impl<F: Float> LassoValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
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
            solver_params: SolverParams::default(),
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
        self.0.solver_params.max_iterations = max_iterations;
        self
    }

    /// Set the maximum number of epochs in the inner loop during the descent
    /// routine.
    /// Defaults to `1000` if not set.
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.0.solver_params.max_epochs = max_epochs;
        self
    }

    /// Set the initial working set size.
    ///
    /// Defaults to `10` if not set.
    pub fn p0(mut self, p0: usize) -> Self {
        self.0.solver_params.p0 = p0;
        self
    }

    /// Set the stopping criterion for the optimization routine (KKT violation).
    ///
    /// Defaults to `1e-6` if not set.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.solver_params.tolerance = tolerance;
        self
    }

    /// Set the number of points used for extrapolation.
    ///
    /// Defaults to `5` if not set.
    pub fn K(mut self, K: usize) -> Self {
        self.0.solver_params.K = K;
        self
    }

    /// Enables the use of Anderson acceleration for the extrapolation of
    /// primal iterates.
    /// Defaults to `true` if not set.
    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.0.solver_params.use_acceleration = use_acceleration;
        self
    }

    /// Sets the verbosity level of the solver.
    ///
    /// Defaults to `true` if not set.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.solver_params.verbose = verbose;
        self
    }
}

impl<F: Float> ParamGuard for LassoParams<F> {
    type Checked = LassoValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        self.0.solver_params.check_ref()?;
        Ok(self.0)
    }
}

/// A verified hyperparameter set ready for the fitting of a MCP regression model
///

pub struct MCPValidParams<F> {
    alpha: F,
    gamma: F,
    solver_params: SolverParams<F>,
}

impl<F: Float> MCPValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }
}

/// A hyper-parameter set during construction
///
/// Configures and minimizes the following objective function:
/// ```ignore
/// 1 / (2 * n_samples) * ||y - Xw||^2_2
///     + alpha * MCP(w)
/// ```
///
pub struct MCParams<F>(MCPValidParams<F>);

impl<F: Float> Default for MCParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a MCP model
impl<F: Float> MCParams<F> {
    /// Create default MCP hyper parameters
    pub fn new() -> MCParams<F> {
        Self(MCPValidParams {
            alpha: F::one(),
            gamma: F::cast(2),
            solver_params: SolverParams::default(),
        })
    }

    /// Set the regularization hyperparameter. A higher value yields sparser
    /// solutions.
    /// Defaults to `1` if not set.
    pub fn alpha(mut self, alpha: F) -> Self {
        self.0.alpha = alpha;
        self
    }

    /// Set the shape of the penalty. A value close to infinity yields is equivalent
    /// soft-thresholding (L1-norm).
    /// Defaults to `2` if not set.
    pub fn gamma(mut self, gamma: F) -> Self {
        self.0.gamma = gamma;
        self
    }

    /// Set the maximum number of iterations in the outer loop used to build
    /// working set.
    /// Defaults to `50` if not set.
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.0.solver_params.max_iterations = max_iterations;
        self
    }

    /// Set the maximum number of epochs in the inner loop during the descent
    /// routine.
    /// Defaults to `1000` if not set.
    pub fn max_epochs(mut self, max_epochs: usize) -> Self {
        self.0.solver_params.max_epochs = max_epochs;
        self
    }

    /// Set the initial working set size.
    ///
    /// Defaults to `10` if not set.
    pub fn p0(mut self, p0: usize) -> Self {
        self.0.solver_params.p0 = p0;
        self
    }

    /// Set the stopping criterion for the optimization routine (KKT violation).
    ///
    /// Defaults to `1e-6` if not set.
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.0.solver_params.tolerance = tolerance;
        self
    }

    /// Set the number of points used for extrapolation.
    ///
    /// Defaults to `5` if not set.
    pub fn K(mut self, K: usize) -> Self {
        self.0.solver_params.K = K;
        self
    }

    /// Enables the use of Anderson acceleration for the extrapolation of
    /// primal iterates.
    /// Defaults to `true` if not set.
    pub fn use_acceleration(mut self, use_acceleration: bool) -> Self {
        self.0.solver_params.use_acceleration = use_acceleration;
        self
    }

    /// Sets the verbosity level of the solver.
    ///
    /// Defaults to `true` if not set.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.0.solver_params.verbose = verbose;
        self
    }
}

impl<F: Float> ParamGuard for MCParams<F> {
    type Checked = MCPValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyperparameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.alpha < F::one() {
            Err(EstimatorError::InvalidGamma(self.0.gamma.to_f32().unwrap()))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        self.0.solver_params.check_ref()?;
        Ok(self.0)
    }
}
