from distributed.utils_test import gen_cluster
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification as sk_make_classification
from sklearn.base import clone
from dask_ml.model_selection import BlackBox, IncrementalSearchCV


def test_black_box_basic():
    params = {"C": np.logspace(-3, 1, num=400)}
    est = LogisticRegression(solver="lbfgs")
    wrapper = BlackBox(est, max_iter=2)
    assert wrapper.max_iter == 2

    X, y = sk_make_classification(n_samples=20, n_features=5)
    wrapper.partial_fit(X, y)
    wrapper.partial_fit(X, y)
    assert wrapper._calls == 2


@gen_cluster(client=True, timeout=5000)
def test_black_box_w_search(c, s, a, b):
    max_iter = 4
    params = {"C": np.logspace(-3, 1, num=400)}
    est = LogisticRegression(solver="lbfgs")
    wrapper = BlackBox(est, max_iter=max_iter)
    X, y = sk_make_classification(n_samples=50, n_features=5)

    w2 = clone(wrapper)
    assert type(w2) == type(wrapper)
    assert type(w2.set_params(C=2 * np.pi)) == type(w2)

    search = IncrementalSearchCV(wrapper, params, decay_rate=0, max_iter=max_iter)
    yield search.fit(X, y)
    assert search.best_estimator_._calls == max_iter
    assert search.best_score_ > 0
    assert type(search.best_estimator_) == type(wrapper)
