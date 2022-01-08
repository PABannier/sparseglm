import time
import numpy as np

from sklearn.linear_model import Lasso as Lasso_sk
from rustylasso.estimators import Lasso

from rustylasso.utils import make_correlated_data, compute_alpha_max

n_samples = 3000
n_features = 10_000

snr = 2
corr = 0.6
density = 0.1

tol = 1e-12

reg = 0.01

X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, corr=corr, snr=snr,
    density=density, random_state=0)


def time_estimator(clf, X, y):
    start = time.time()
    clf.fit(X, y)
    duration = time.time() - start
    return clf.coef_, duration


alpha_max = compute_alpha_max(X, y)

estimator_sk = Lasso_sk(alpha_max * reg, fit_intercept=False, tol=tol)
estimator_rl = Lasso(alpha_max * reg, tol=tol, verbose=True)

coef_sk, duration_sk = time_estimator(estimator_sk, X, y)
coef_rl, duration_rl = time_estimator(estimator_rl, X, y)

np.testing.assert_allclose(coef_sk, coef_rl, atol=1e-6)

print(f"Scikit-learn :: {duration_sk} s")
print(f"RustyLasso :: {duration_rl} s")
