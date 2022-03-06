import pytest

import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp

from sklearn.linear_model import Lasso as Lasso_sk
from sklearn.linear_model import ElasticNet as ElasticNet_sk
from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sk
from sklearn.linear_model import MultiTaskElasticNet as MultiTaskElasticNet_sk

from sparseglm.estimators import (Lasso, MultiTaskLasso, ElasticNet, 
                                  MultiTaskElasticNet)

from ..utils import make_correlated_data


n_samples = 50
n_tasks = 30
n_features = 60

X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.1,
    random_state=0)
y = Y[:, 0]

X_sparse = sp.csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

np.random.seed(0)

alpha_max = np.max(norm(X.T @ Y, axis=1, ord=2)) / n_samples
alpha = 0.05 * alpha_max

l1_ratio = 0.4

tol = 1e-10


@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('type', [np.float64])
def test_lasso(X, type):
    clf = Lasso(alpha, tol=tol)
    clf_sk = Lasso_sk(alpha, tol=tol, fit_intercept=False)

    X_conv = X.astype(type)
    y_conv = y.astype(type)

    clf.fit(X_conv, y_conv)
    clf_sk.fit(X_conv, y_conv)

    coef = clf.coef_
    coef_sk = clf_sk.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)


@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('type', [np.float64])
def test_elastic_net(X, type):
    clf = ElasticNet(alpha, l1_ratio, tol=tol)
    clf_sk = ElasticNet_sk(alpha, l1_ratio=l1_ratio, tol=tol, 
                           fit_intercept=False)

    X_conv = X.astype(type)
    y_conv = y.astype(type)

    clf.fit(X_conv, y_conv)
    clf_sk.fit(X_conv, y_conv)

    coef = clf.coef_
    coef_sk = clf_sk.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)


@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('type', [np.float64])
def test_multitasklasso(X, type):
    clf = MultiTaskLasso(alpha, tol=tol)
    clf_sk = MultiTaskLasso_sk(alpha, tol=tol, fit_intercept=False)

    X_conv = X.astype(type)
    Y_conv = Y.astype(type)

    clf.fit(X_conv, Y_conv)
    clf_sk.fit(X_conv.toarray() if sp.issparse(X_conv) else X_conv, Y_conv)

    coef = clf.coef_
    coef_sk = clf_sk.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)


@pytest.mark.parametrize('X', [X, X_sparse])
@pytest.mark.parametrize('type', [np.float64])
def test_multitaskelasticnet(X, type):
    clf = MultiTaskElasticNet(alpha, l1_ratio, tol=tol)
    clf_sk = MultiTaskElasticNet_sk(alpha, l1_ratio=l1_ratio, tol=tol,
                                    fit_intercept=False)

    X_conv = X.astype(type)
    Y_conv = Y.astype(type)

    clf.fit(X_conv, Y_conv)
    clf_sk.fit(X_conv.toarray() if sp.issparse(X_conv) else X_conv, Y_conv)

    coef = clf.coef_
    coef_sk = clf_sk.coef_

    np.testing.assert_allclose(coef, coef_sk, atol=1e-6)
