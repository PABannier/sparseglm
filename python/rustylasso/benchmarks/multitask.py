import time

import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import MultiTaskLasso as MultiTaskLasso_sk
from rustylasso.estimators import MultiTaskLasso

from rustylasso.utils import make_correlated_data, compute_alpha_max

n_samples = 100
n_features = 1000
n_tasks = 30

snr = 2
corr = 0.6
density = 0.1

tol = 1e-9

reg = 0.05

X, Y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, 
    corr=corr, snr=snr, density=density, random_state=0)

X_sparse = sp.csc_matrix(X * np.random.binomial(1, 0.1, X.shape))


def time_estimator(clf, X, y):
    start = time.time()
    clf.fit(X, y)
    duration = time.time() - start
    return clf.coef_, duration


alpha_max = compute_alpha_max(X, Y)

estimator_sk = MultiTaskLasso_sk(alpha_max * reg, fit_intercept=False, tol=tol,
                                 max_iter=10**6)
estimator_rl = MultiTaskLasso(alpha_max * reg, tol=tol, verbose=False, 
                              max_epochs=100_000)

print("Fitting dense matrices...")

coef_sk, duration_sk = time_estimator(estimator_sk, X, Y)
coef_rl, duration_rl = time_estimator(estimator_rl, X, Y)

np.testing.assert_allclose(coef_sk, coef_rl, atol=1e-5)

print("Fitting sparse matrices...")

coef_sk_sparse, duration_sk_sparse = time_estimator(
    estimator_sk, X_sparse.toarray(), Y
)
coef_rl_sparse, duration_rl_sparse = time_estimator(estimator_rl, X_sparse, Y)

np.testing.assert_allclose(coef_sk_sparse, coef_rl_sparse, atol=1e-5)

print("=" * 5 + " RESULTS " + "=" * 5)

print(f"[DENSE] Scikit-learn :: {duration_sk} s")
print(f"[DENSE] RustyLasso :: {duration_rl} s")
print("--" * 5)
print(f"[SPARSE] Scikit-learn :: {duration_sk_sparse} s")
print(f"[SPARSE] RustyLasso :: {duration_rl_sparse} s")
