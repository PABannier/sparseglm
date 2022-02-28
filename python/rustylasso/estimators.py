import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator
from sklearn.utils import check_array

import rustylassopy


__all__ = ["Lasso", "MultiTaskLasso", "MCPRegressor", "BlockMCPRegressor"]


class Estimator(BaseEstimator):
    r"""Base estimator class for common initialization."""
    def __init__(self, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.tol = tol
        self.p0 = p0
        self.use_accel = use_accel
        self.K = K
        self.verbose = verbose

        self.coef_ = None

    def _validate_params(self):
        def _check_types(var, type, var_name):
            if not isinstance(var, type) or var < 0:
                raise ValueError("{}={} must be a positive {}."
                                 .format(var_name, self.alpha, type))
        _check_types(self.K, int, "K")
        _check_types(self.p0, int, "p0")
        _check_types(self.tol, float, "tol")
        _check_types(self.max_iter, int, "max_iter")
        _check_types(self.max_epochs, int, "max_epochs")

    def predict(self, X):
        """Predicts the test set."""
        if not self.coef_:
            raise Exception("Estimator must be fitted before inference.")
        return X @ self.coef_.T


class Lasso(Estimator):
    r"""Solves a L1-regularized least square linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------
    alpha : float
        The regularization hyperparameter.
    
    max_iter : int, default = 50
        The number of iterations of the outer CD solver.
    
    max_epochs : int, default = 1000
        The number of iterations used by the inner CD solver.
    
    tolerance : float, default = 1e-9
        Stopping criterion used.
    
    p0 : int, default = 10
        The starting size of the working set.
    
    use_accel : bool, default = True
        Usage of Anderson acceleration.
    
    K : int, default = 5
        Number of primal points used to extrapolate.
    
    verbose : bool, default = True
        Verbosity.

    Examples
    --------
    >>> from rustylasso.estimators import Lasso
    >>> clf = Lasso(alpha)
    >>> clf.fit(X, y)
    """
    def __init__(self, alpha, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        super(Lasso, self).__init__(
            max_iter=max_iter, tol=tol, p0=p0, max_epochs=max_epochs, K=K,
            use_accel=use_accel, verbose=verbose)
        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("alpha={} must be a positive float".format(alpha))
        else:
            self.alpha = alpha

    def fit(self, X, y):
        r"""Solves the L1-regularized linear regression to the data (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        y : array-like, shape (n_samples)
            Measurements.
        """
        self._validate_params()
        X = check_array(X, 'csc', dtype=[np.float64, np.float32], order='F',
                        copy=False, accept_large_sparse=False)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)

        self._inner = rustylassopy.LassoWrapper(
            alpha=self.alpha, max_iterations=self.max_iter, p0=self.p0,
            k=self.K, max_epochs=self.max_epochs, tolerance=self.tol,
            use_acceleration=self.use_accel, verbose=self.verbose)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, y)
        else:
            coefs = self._inner.fit(X, y)

        self.coef_ = coefs.T
        return self


class MultiTaskLasso(Estimator):
    r"""Solves a L21-regularized least square multi-task linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------
    alpha : float
        The regularization hyperparameter.
    
    max_iter : int, default = 50
        The number of iterations of the outer CD solver.
    
    max_epochs : int, default = 1000
        The number of iterations used by the inner CD solver.
    
    tolerance : float, default = 1e-9
        Stopping criterion used.
    
    p0 : int, default = 10
        The starting size of the working set.
    
    use_accel : bool, default = True
        Usage of Anderson acceleration.
    
    K : int, default = 5
        Number of primal points used to extrapolate.
    
    verbose : bool, default = True
        Verbosity.

    Examples
    --------
    >>> from rustylasso.estimators import MultiTaskLasso
    >>> clf = MultiTaskLasso(alpha)
    >>> clf.fit(X, Y)

    """
    def __init__(self, alpha, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        super(MultiTaskLasso, self).__init__(
            max_iter=max_iter, tol=tol, p0=p0, K=K, verbose=verbose,
            use_accel=use_accel, max_epochs=max_epochs)
        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("alpha={} must be a positive float".format(alpha))
        else:
            self.alpha = alpha

    def fit(self, X, Y):
        r"""Solves the L21-regularized linear regression to the data (X, Y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        Y : array-like, shape (n_samples, n_tasks)
            Measurements.
        """
        self._validate_params()
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              accept_sparse='csc')
        check_Y_params = dict(ensure_2d=False, order='F')
        X, Y = self._validate_data(X, Y, validate_separately=(check_X_params,
                                                              check_Y_params))
        Y = Y.astype(X.dtype)

        if Y.ndim == 1:
            raise ValueError("For mono-task outputs, use Lasso")

        n_samples = X.shape[0]

        if n_samples != Y.shape[0]:
            raise ValueError("X and Y have inconsistent dimensions (%d != %d)"
                             % (n_samples, Y.shape[0]))

        self._inner = rustylassopy.MultiTaskLassoWrapper(
            alpha=self.alpha, max_iterations=self.max_iter, p0=self.p0,
            k=self.K, max_epochs=self.max_epochs, tolerance=self.tol,
            use_acceleration=self.use_accel, verbose=self.verbose)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, Y)
        else:
            coefs = self._inner.fit(X, Y)

        self.coef_ = coefs.T
        return self


class MCPRegressor(Estimator):
    r"""Solves a MCP-regularized least square linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------
    alpha : float
        The regularization hyperparameter.

    gamma : float
        The shaping hyperparameter of the MCP. gamma=`np.inf` corresponds to a 
        soft-thresholding operator.
    
    max_iter : int, default = 50
        The number of iterations of the outer CD solver.
    
    max_epochs : int, default = 1000
        The number of iterations used by the inner CD solver.
    
    tolerance : float, default = 1e-9
        Stopping criterion used.
    
    p0 : int, default = 10
        The starting size of the working set.
    
    use_accel : bool, default = True
        Usage of Anderson acceleration.
    
    K : int, default = 5
        Number of primal points used to extrapolate.
    
    verbose : bool, default = True
        Verbosity.

    Examples
    --------
    >>> from rustylasso.estimators import MCPRegressor
    >>> clf = MCP(alpha, gamma)
    >>> clf.fit(X, y)

    """
    def __init__(self, alpha, gamma, max_iter=50, max_epochs=1000, tol=1e-9,
                 p0=10, use_accel=True, K=5, verbose=True):
        super(MCPRegressor, self).__init__(max_iter=max_iter, K=K, tol=tol,
                                           p0=p0, max_epochs=max_epochs,
                                           use_accel=use_accel, verbose=verbose)
        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("alpha={} must be a positive float".format(alpha))
        elif not isinstance(gamma, float) or alpha < 1:
            raise ValueError("gamma={} must be a float greater than \
                              1".format(gamma))
        else:
            self.alpha = alpha
            self.gamma = gamma

    def fit(self, X, y):
        r"""Solves the L1-regularized linear regression to the data (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        y : array-like, shape (n_samples)
            Measurements.
        """
        self._validate_params()
        X = check_array(X, 'csc', dtype=[np.float64, np.float32], order='F',
                        copy=False, accept_large_sparse=False)
        y = check_array(y, 'csc', dtype=X.dtype.type, order='F', copy=False,
                        ensure_2d=False)

        self._inner = rustylassopy.MCPWrapper(
            alpha=self.alpha, gamma=self.gamma, max_iterations=self.max_iter,
            p0=self.p0, k=self.K, max_epochs=self.max_epochs,
            tolerance=self.tol, use_acceleration=self.use_accel,
            verbose=self.verbose)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, y)
        else:
            coefs = self._inner.fit(X, y)

        self.coef_ = coefs.T
        return self


class BlockMCPRegressor(Estimator):
    r"""Solves a BlockMCP-regularized least square multi-task linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------
    alpha : float
        The regularization hyperparameter.

    gamma : float
        The shaping hyperparameter of the MCP. gamma=`np.inf` corresponds to a 
        soft-thresholding operator.
    
    max_iter : int, default = 50
        The number of iterations of the outer CD solver.
    
    max_epochs : int, default = 1000
        The number of iterations used by the inner CD solver.
    
    tolerance : float, default = 1e-9
        Stopping criterion used.
    
    p0 : int, default = 10
        The starting size of the working set.
    
    use_accel : bool, default = True
        Usage of Anderson acceleration.
    
    K : int, default = 5
        Number of primal points used to extrapolate.
    
    verbose : bool, default = True
        Verbosity.

    Examples
    --------
    >>> from rustylasso.estimators import BlockMCPRegressor
    >>> clf = BlockMCPRegressor(alpha, gamma)
    >>> clf.fit(X, Y)

    """
    def __init__(self, alpha, gamma, max_iter=50, max_epochs=1000, tol=1e-9,
                 p0=10, use_accel=True, K=5, verbose=True):
        super(BlockMCPRegressor, self).__init__(
            max_iter=max_iter, tol=tol, p0=p0, K=K, verbose=verbose,
            use_accel=use_accel, max_epochs=max_epochs)
        if not isinstance(alpha, float) or alpha < 0:
            raise ValueError("alpha={} must be a positive float".format(alpha))
        elif not isinstance(gamma, float) or alpha < 1:
            raise ValueError("gamma={} must be a float greater than \
                              1".format(gamma))
        else:
            self.alpha = alpha
            self.gamma = gamma

    def fit(self, X, Y):
        r"""Solves the BlockMCP-regularized linear regression to the data (X, Y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.

        Y : array-like, shape (n_samples, n_tasks)
            Measurements.
        """
        self._validate_params()
        check_X_params = dict(dtype=[np.float64, np.float32], order='F',
                              accept_sparse='csc')
        check_Y_params = dict(ensure_2d=False, order='F')
        X, Y = self._validate_data(X, Y, validate_separately=(check_X_params,
                                                              check_Y_params))
        Y = Y.astype(X.dtype)

        if Y.ndim == 1:
            raise ValueError("For mono-task outputs, use MCP")

        n_samples = X.shape[0]

        if n_samples != Y.shape[0]:
            raise ValueError("X and Y have inconsistent dimensions (%d != %d)"
                             % (n_samples, Y.shape[0]))

        self._inner = rustylassopy.BlockMCPWrapper(
            alpha=self.alpha, gamma=self.gamma, max_iterations=self.max_iter,
            p0=self.p0, k=self.K, max_epochs=self.max_epochs,
            tolerance=self.tol, use_acceleration=self.use_accel,
            verbose=self.verbose)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, Y)
        else:
            coefs = self._inner.fit(X, Y)

        self.coef_ = coefs.T
        return self
