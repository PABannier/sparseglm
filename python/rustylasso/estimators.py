import rustylassopy

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y


__all__ = ["Lasso"]


class Lasso(BaseEstimator):
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

        _check_types(self.alpha, float, "alpha")
        _check_types(self.max_iter, int, "max_iter")
        _check_types(self.max_epochs, int, "max_epochs")
        _check_types(self.tol, float, "tol")
        _check_types(self.p0, int, "p0")
        _check_types(self.K, int, "K")

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
        X, y = check_X_y(X, y, accept_sparse='csc', order="C")

        if y.dtype == np.float32 and X.dtype == np.float32:
            self._inner = rustylassopy.LassoWrapperF32(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)
        elif y.dtype == np.float64 and X.dtype == np.float64:
            self._inner = rustylassopy.LassoWrapperF64(
                alpha=self.alpha, max_iter=self.max_iter, p0=self.p0, K=self.K,
                max_epochs=self.max_epochs, tol=self.tol, verbose=self.verbose,
                use_accel=self.use_accel)
        else:
            raise TypeError("X and y must have the same type. Got {} and {}"
                            .format(X.data.dtype, y.dtype))

        if sp.issparse(X):
            coefs = self._inner.fit_sparse(X.data, X.indices, X.indptr, y)
        else:
            coefs = self._inner.fit(X, y)

        self.coef_ = coefs.T
        return self

    def predict(self, X):
        """Predits the test set."""
        # TODO: validation of X
        return X @ self.coef_.T
