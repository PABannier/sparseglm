import pytest

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from sklearn.linear_model import Lasso as Lasso_sk
from rustylasso.estimators import Lasso

from .utils import make_correlated_data

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
def test_estimator(X):
    clf = Lasso(alpha, tol=tol)
    clf_sk = Lasso_sk(alpha, tol=tol)

    clf.fit(X, y)
    clf_sk.fit(X, y)
    coef = clf.coef_
    coef_sk = clf.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)
