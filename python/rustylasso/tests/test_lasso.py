import pytest

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from sklearn.linear_model import Lasso as Lasso_sk
from rustylasso.estimators import Lasso

from ..utils import make_correlated_data

n_samples = 50
n_tasks = 1
n_features = 60
X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.1,
    random_state=0)

X_sparse = sp.csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

np.random.seed(0)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max

tol = 1e-10


@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('type', [np.float32, np.float64])
def test_estimator(X, type):
    clf = Lasso(alpha, tol=tol)
    clf_sk = Lasso_sk(alpha, tol=tol, fit_intercept=False)

    X_conv = X.astype(type)
    y_conv = y.astype(type)

    clf_sk.fit(X_conv, y_conv)
    clf.fit(X_conv, y_conv)
    coef = clf.coef_
    coef_sk = clf.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)
