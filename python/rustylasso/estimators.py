import rustylassopy

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.utils import check_array


__all__ = ["Lasso", "MultiTaskLasso"]


class Estimator(BaseEstimator):
    """Base estimator class for common initialization."""
    def __init__(self, alpha, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        self.alpha = alpha
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
        _check_types(self.alpha, float, "alpha")
        _check_types(self.max_iter, int, "max_iter")
        _check_types(self.max_epochs, int, "max_epochs")

    def predict(self, X):
        """Predicts the test set."""
        if not self.coef_:
            raise Exception("Estimator must be fitted before inference.")
        return X @ self.coef_.T


class Lasso(Estimator):
    """Solves a L1-regularized least square linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------

    TODO

    Examples
    --------
    >>> from rustylasso.estimators import Lasso
    >>> clf = Lasso(alpha)
    >>> clf.fit(X, y)

    """
    def __init__(self, alpha, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        super(Lasso, self).__init__(alpha=alpha, max_iter=max_iter, tol=tol,
                                    p0=p0, max_epochs=max_epochs, K=K,
                                    use_accel=use_accel, verbose=verbose)

    def fit(self, X, y):
        """Solves the L1-regularized linear regression to the data (X, y).

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

        if (X.dtype, y.dtype) == (np.float32, np.float32):
            self._inner = rustylassopy.LassoWrapperF32(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)
        else:
            self._inner = rustylassopy.LassoWrapperF64(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, y)
        else:
            coefs = self._inner.fit(X, y)

        self.coef_ = coefs.T
        return self


class MultiTaskLasso(Estimator):
    """Solves a L21-regularized least square multi-task linear regression.

    The solver uses Anderson acceleration combined with a working set strategy
    for faster convergence.

    Parameters
    ----------

    TODO

    Examples
    --------
    >>> from rustylasso.estimators import MultiTaskLasso
    >>> clf = MultiTaskLasso(alpha)
    >>> clf.fit(X, Y)

    """
    def __init__(self, alpha, max_iter=50, max_epochs=1000, tol=1e-9, p0=10,
                 use_accel=True, K=5, verbose=True):
        super(MultiTaskLasso, self).__init__(
            alpha=alpha, max_iter=max_iter, tol=tol, p0=p0, K=K,
            verbose=verbose, use_accel=use_accel, max_epochs=max_epochs)

    def fit(self, X, Y):
        """Solves the L21-regularized linear regression to the data (X, Y).

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

        if (X.dtype, Y.dtype) == (np.float32, np.float32):
            self._inner = rustylassopy.MultiTaskLassoWrapperF32(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)
        else:
            self._inner = rustylassopy.MultiTaskLassoWrapperF64(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, Y)
        else:
            coefs = self._inner.fit(X, Y)

        self.coef_ = coefs.T
        return self
