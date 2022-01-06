import time

import numpy as np
import scipy.sparse as sp

from sklearn.linear_model import Lasso as Lasso_sk
from rustylassopy.estimators import Lasso
from .utils import make_correlated_data

n_samples = 300
n_tasks = 1
n_features = 1000
X, y, _ = make_correlated_data(
    n_samples=n_samples, n_features=n_features, n_tasks=n_tasks, density=0.1,
    random_state=0)

X_sparse = sp.csc_matrix(X * np.random.binomial(1, 0.1, X.shape))

np.random.seed(0)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.05 * alpha_max

tol = 1e-10

durations = {}

print("Benchmarking...")

clf_sk = Lasso_sk(alpha, tol=tol)
start = time.time()
clf_sk.fit(X, y)
duration_sk_dense = time.time() - start

start = time.time()
clf_sk.fit(X_sparse, y)
duration_sk_sparse = time.time() - start

clf = Lasso(alpha, tol=tol)
start = time.time()
clf.fit(X, y)
duration_rl_dense = time.time() - start

start = time.time()
clf.fit(X_sparse, y)
duration_rl_sparse = time.time() - start


print("=" * 20)

print("Scikit-Learn | Dense ::", duration_sk_dense)
print("Scikit-Learn | Sparse ::", duration_sk_sparse)
print("RustyLasso | Dense :: ", duration_rl_dense)
print("RustyLasso | Sparse ::", duration_rl_sparse)

