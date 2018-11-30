import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin


class BlackBox(BaseEstimator, MetaEstimatorMixin):
    """
    Fit an estimator with a subset of the data passed.

    This is most useful in cross validation searches because it treats the
    estimator as a black box.

    Parameters
    ----------
    est : BaseEstimator
        The base estimator to fit with data

    max_calls : int, optional
        The maximum number of times this estimator wil be called.

    """

    def __init__(self, estimator, max_iter=1, **kwargs):
        self.estimator = estimator
        self.max_iter = max_iter
        self.kwargs = kwargs
        self._calls = 0

    def _set_kwargs(self):
        est_kwargs = {k[4:]: v for k, v in self.kwargs.items() if k[:4] == "est_"}
        return self.estimator.set_params(**est_kwargs)

    def fit(self, X, y):
        self.estimator = self._set_kwargs()
        self.estimator.fit(X, y)
        return self

    def partial_fit(self, X, y=None):
        if self._calls == 0:
            self.estimator = self._set_kwargs()
        self._calls += 1
        frac = self._calls / self.max_iter
        n = X.shape[0]
        num_data = int(n * frac)
        num_data = np.clip(num_data, 2, n)
        idx = np.random.permutation(n)[:num_data]
        self.estimator.fit(X[idx], y[idx])
        return self

    def score(self, X, y=None):
        return self.estimator.score(X, y)

    def get_params(self, **kwargs):
        est_kwargs = {
            "est_" + k: v for k, v in self.estimator.get_params(**kwargs).items()
        }
        return {"estimator": self.estimator, **est_kwargs}

    def set_params(self, **kwargs):
        est_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == "est_"}
        other_kwargs = {k: v for k, v in kwargs.items() if k[4:] not in est_kwargs}
        self.estimator.set_params(**est_kwargs)
        for k, v in other_kwargs.items():
            setattr(self, k, v)
        return self
