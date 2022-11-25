use super::error::{EstimatorError, Result};
use super::param_guard::ParamGuard;
use super::Float;

/// A verified hyperparameter set ready for the fitting of a Lasso regression model
#[derive(Debug, Clone, PartialEq)]
pub struct LassoValidParams<F> {
    alpha: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
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

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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

#[derive(Debug, Clone, PartialEq)]
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
            ws_start_size: 10,
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.ws_start_size <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
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
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskLassoValidParams<F> {
    alpha: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
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

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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

#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskLassoParams<F>(MultiTaskLassoValidParams<F>);

impl<F: Float> Default for MultiTaskLassoParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a MultiTaskLasso model
impl<F: Float> MultiTaskLassoParams<F> {
    /// Create default MultiTaskLasso hyper parameters
    pub fn new() -> MultiTaskLassoParams<F> {
        Self(MultiTaskLassoValidParams {
            alpha: F::one(),
            max_iterations: 50,
            max_epochs: 1000,
            ws_start_size: 10,
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.ws_start_size <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// A verified hyperparameter set ready for the fitting of a MCP regression model
#[derive(Debug, Clone, PartialEq)]
pub struct MCPValidParams<F> {
    alpha: F,
    gamma: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> MCPValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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
///     + alpha * MCP(w)
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct MCParams<F>(MCPValidParams<F>);

impl<F: Float> Default for MCParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a Lasso model
impl<F: Float> MCParams<F> {
    /// Create default Lasso hyper parameters
    pub fn new() -> MCParams<F> {
        Self(MCPValidParams {
            alpha: F::one(),
            gamma: F::cast(3.),
            max_iterations: 50,
            max_epochs: 1000,
            ws_start_size: 10,
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

    /// Set the shape of the concave penalty.
    /// Defaults to `3` if not set.
    pub fn gamma(mut self, gamma: F) -> Self {
        self.0.gamma = gamma;
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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

impl<F: Float> ParamGuard for MCParams<F> {
    type Checked = MCPValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.gamma < F::one() {
            Err(EstimatorError::InvalidGamma(self.0.gamma.to_f32().unwrap()))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.ws_start_size <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// Configure a BlockMCP model.

#[derive(Debug, Clone, PartialEq)]
pub struct BlockMCPValidParams<F> {
    alpha: F,
    gamma: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> BlockMCPValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn gamma(&self) -> F {
        self.gamma
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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

#[derive(Debug, Clone, PartialEq)]
pub struct BlockMCParams<F>(BlockMCPValidParams<F>);

impl<F: Float> Default for BlockMCParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a Lasso model
impl<F: Float> BlockMCParams<F> {
    /// Create default Lasso hyper parameters
    pub fn new() -> BlockMCParams<F> {
        Self(BlockMCPValidParams {
            alpha: F::one(),
            gamma: F::cast(3),
            max_iterations: 50,
            max_epochs: 1000,
            ws_start_size: 10,
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

    /// Set the shape of the penalty.
    /// Defaults to `3` if not set.
    pub fn gamma(mut self, gamma: F) -> Self {
        self.0.gamma = gamma;
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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

impl<F: Float> ParamGuard for BlockMCParams<F> {
    type Checked = BlockMCPValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.p0 <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// Configure an ElasticNet model.

#[derive(Debug, Clone, PartialEq)]
pub struct ElasticNetValidParams<F> {
    alpha: F,
    l1_ratio: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> ElasticNetValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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
///     + alpha * l1_ratio * ||w||_1
///     + alpha * 0.5 * (1 - l1_ratio) * ||w||_2^2
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticNetParams<F>(ElasticNetValidParams<F>);

impl<F: Float> Default for ElasticNetParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a ElasticNet model
impl<F: Float> ElasticNetParams<F> {
    /// Create default ElasticNet hyper parameters
    pub fn new() -> ElasticNetParams<F> {
        Self(ElasticNetValidParams {
            alpha: F::one(),
            l1_ratio: F::cast(0.5),
            max_iterations: 50,
            max_epochs: 1000,
            ws_start_size: 10,
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

    /// Set the weighting between L1 and L2 penalty.
    /// Defaults to `3` if not set.
    pub fn l1_ratio(mut self, l1_ratio: F) -> Self {
        self.0.l1_ratio = l1_ratio;
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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

impl<F: Float> ParamGuard for ElasticNetParams<F> {
    type Checked = ElasticNetValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.l1_ratio > F::one() || self.0.l1_ratio < F::zero() {
            Err(EstimatorError::InvalidL1Ratio(
                self.0.l1_ratio.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.p0 <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// Configure a MultiTaskElasticNet model.

#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskElasticNetValidParams<F> {
    alpha: F,
    l1_ratio: F,
    max_iterations: usize,
    max_epochs: usize,
    ws_start_size: usize,
    tolerance: F,
    K: usize,
    use_acceleration: bool,
    verbose: bool,
}

impl<F: Float> MultiTaskElasticNetValidParams<F> {
    pub fn alpha(&self) -> F {
        self.alpha
    }

    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    pub fn max_epochs(&self) -> usize {
        self.max_epochs
    }

    pub fn ws_start_size(&self) -> usize {
        self.ws_start_size
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
///     + alpha * l1_ratio * ||w||_1
///     + alpha * 0.5 * (1 - l1_ratio) * ||w||_2^2
/// ```
///
#[derive(Debug, Clone, PartialEq)]
pub struct MultiTaskElasticNetParams<F>(MultiTaskElasticNetValidParams<F>);

impl<F: Float> Default for MultiTaskElasticNetParams<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configure and fit a ElasticNet model
impl<F: Float> MultiTaskElasticNetParams<F> {
    /// Create default ElasticNet hyper parameters
    pub fn new() -> MultiTaskElasticNetParams<F> {
        Self(MultiTaskElasticNetValidParams {
            alpha: F::one(),
            l1_ratio: F::cast(0.5),
            max_iterations: 50,
            max_epochs: 1000,
            ws_start_size: 10,
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

    /// Set the weighting between L1 and L2 penalty.
    /// Defaults to `3` if not set.
    pub fn l1_ratio(mut self, l1_ratio: F) -> Self {
        self.0.l1_ratio = l1_ratio;
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
    pub fn ws_start_size(mut self, ws_start_size: usize) -> Self {
        self.0.ws_start_size = ws_start_size;
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

impl<F: Float> ParamGuard for MultiTaskElasticNetParams<F> {
    type Checked = MultiTaskElasticNetValidParams<F>;
    type Error = EstimatorError;

    /// Validate the hyper parameters
    fn check_ref(&self) -> Result<&Self::Checked> {
        if self.0.alpha.is_negative() {
            Err(EstimatorError::InvalidRegularization(
                self.0.alpha.to_f32().unwrap(),
            ))
        } else if self.0.l1_ratio > F::one() || self.0.l1_ratio < F::zero() {
            Err(EstimatorError::InvalidL1Ratio(
                self.0.l1_ratio.to_f32().unwrap(),
            ))
        } else if self.0.tolerance.is_negative() {
            Err(EstimatorError::InvalidTolerance(
                self.0.tolerance.to_f32().unwrap(),
            ))
        } else if self.0.K <= 0 {
            Err(EstimatorError::InvalidK(self.0.K))
        } else if self.0.p0 <= 0 {
            Err(EstimatorError::InvalidP0(self.0.p0))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked> {
        self.check_ref()?;
        Ok(self.0)
    }
}
