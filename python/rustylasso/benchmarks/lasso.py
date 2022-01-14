import time

import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import Lasso as Lasso_sk
from rustylasso.estimators import Lasso

from rustylasso.utils import make_correlated_data, compute_alpha_max

n_samples = 3_000
n_features = 10_000

snr = 2
corr = 0.6
density = 0.5

tol = 1e-9

reg = 0.05

X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, corr=corr, snr=snr,
    density=density, random_state=0)

X_sparse = sp.csc_matrix(X * np.random.binomial(1, 0.1, X.shape))


def time_estimator(clf, X, y):
    start = time.time()
    clf.fit(X, y)
    duration = time.time() - start
    return clf.coef_, duration


alpha_max = compute_alpha_max(X, y)


estimator_sk = Lasso_sk(alpha_max * reg, fit_intercept=False, tol=tol,
                        max_iter=10**6)
estimator_rl = Lasso(alpha_max * reg, tol=tol, verbose=True)

print("Fitting dense matrices...")

print("=" * 5 + " SCIKIT-LEARN " + 5 * " ")
coef_sk, duration_sk = time_estimator(estimator_sk, X, y)
print("=" * 5 + " RUSTY-LASSO " + 5 * " ")
coef_rl, duration_rl = time_estimator(estimator_rl, X, y)

np.testing.assert_allclose(coef_sk, coef_rl, atol=1e-5)

print("Fitting sparse matrices...")

print("=" * 5 + " SCIKIT-LEARN " + 5 * " ")
coef_sk_sparse, duration_sk_sparse = time_estimator(estimator_sk, X_sparse, y)
print("=" * 5 + " RUSTY-LASSO " + 5 * " ")
coef_rl_sparse, duration_rl_sparse = time_estimator(estimator_rl, X_sparse, y)

np.testing.assert_allclose(coef_sk_sparse, coef_rl_sparse, atol=1e-6)

print("\n")
print("=" * 5 + " RESULTS " + "=" * 5)

print(f"[DENSE] Scikit-learn :: {duration_sk} s")
print(f"[DENSE] RustyLasso :: {duration_rl} s")
print("--" * 5)
print(f"[SPARSE] Scikit-learn :: {duration_sk_sparse} s")
print(f"[SPARSE] RustyLasso :: {duration_rl_sparse} s")
