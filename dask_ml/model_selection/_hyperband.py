from __future__ import division

from copy import deepcopy
import logging
import math

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import ParameterSampler
from sklearn.utils import check_random_state
from sklearn.metrics.scorer import check_scoring
from tornado import gen
import toolz

import dask.array as da
from dask.distributed import as_completed, default_client, futures_of
from distributed.metrics import time

from ._split import train_test_split
from ._search import DaskBaseSearchCV
from ._incremental import fit as _incremental_fit
from ._successive_halving import _SHA



logger = logging.getLogger(__name__)


def _get_hyperband_params(R, eta=3):
    """
    Parameters
    ----------
    R : int
        The maximum number of iterations desired.

    Returns
    -------
    N : list
        The number of models for each bracket
    R : list
        The number of iterations for each bracket
    brackets : list
        The bracket identifier.
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R

    brackets = list(reversed(range(int(s_max + 1))))
    N = [math.ceil(B / R * eta ** s / (s + 1)) for s in brackets]
    R = [int(R * eta ** -s) for s in brackets]
    return list(map(int, N)), R, brackets


def _to_promote(result, completed_jobs, eta=3, asynchronous=True):
    """
    Arguments
    ---------
    result : dict
        with keys ``bracket_iter``, ``bracket``, ``num_models``,
        ``partial_fit_calls``
    completed_jobs : dict values
        List of completed jobs. Each job has the same keys as ``results``
    eta : int
        How aggressive to be in pruning
    asynchronous : bool
        Make a guess of wether to continue with this ``result``?

    Returns
    -------
    jobs : list
        List of jobs to continue training. If asynchronous=True, this is
        result but modified for additional trainig. If asynchronous=False, this
        is a list of results in ``bracket_models`` that need to be further
        trained
    """
    bracket_models = [
        r
        for r in completed_jobs
        if r["bracket_iter"] == result["bracket_iter"]
        and r["bracket"] == result["bracket"]
    ]
    to_keep = len(bracket_models) // eta

    bracket_completed = len(bracket_models) == result["num_models"]
    if not asynchronous and not bracket_completed:
        to_keep = 0
    if to_keep <= 1:
        return []

    def _promote_job(job):
        job["num_models"] = to_keep
        job["partial_fit_calls"] *= eta
        job["bracket_iter"] += 1
        return job

    top = toolz.topk(to_keep, bracket_models, key="score")
    if not asynchronous:
        for job in top:
            job = _promote_job(job)
        return top
    else:
        if to_keep == 0:
            return [result]
        if result in top:
            result = _promote_job(result)
            return [result]
        return []


class HyperbandCV(DaskBaseSearchCV):
    """Find the best parameters for a particular model with cross-validation

    This algorithm is state-of-the-art and only requires computational budget
    as input. It does not require a trade-off between "evaluate many
    parameters" and "train for a long time" like RandomizedSearchCV. Hyperband
    will find close to the best possible parameters with the given
    computational budget [1]_.*

    :sup:`* This will happen with high probability, and "close" means "within
    a log factor of the lower bound"`

    Parameters
    ----------
    model : object
        An object that has support for ``partial_fit``, ``get_params``,
        ``set_params`` and ``score``. This can be an instance of scikit-learn's
        BaseEstimator
    params : dict
        The various parameters to search over.
    max_iter : int, default=81
        The maximum number of partial_fit calls to any one model. This should
        be the number of ``partial_fit`` calls required for the model to
        converge.
    batch_size : float, default=0.2
        The fraction of the dataset that one partial_fit call will see. At
        most, ``batch_size * max_iter`` samples will be seen by
        ``partial_fit``.
    eta : int, default=3
        How aggressive to be in model tuning. It is not recommended to change
        this value, and if changed we recommend ``eta=4``.
        The theory behind Hyperband suggests ``eta=np.e``. Higher
        values imply higher confidence in model selection.
    asynchronous : bool
        Controls the adaptive process by estimating which models to train
        further or making an informed choice by waiting for all jobs to
        complete.  Having many workers benefits or quick jobs benefits from
        asynchronous=True, which is what is recommended.
    random_state : int or np.random.RandomState
        A random state for this class. Setting this helps enforce determinism.
    scoring : str or callable
        The scoring method by which to score different classifiers.
    test_size : float
        Hyperband uses one test set for all example, and this controls the
        size of that test set. It should be a floating point value between 0
        and 1 to represent the number of examples to put into the test set.

    Examples
    --------
    >>> import numpy as np
    >>> from dask_ml.model_selection import HyperbandCV
    >>> from dask_ml.datasets import make_classification
    >>> from sklearn.linear_model import SGDClassifier
    >>>
    >>> X, y = make_classification(chunks=20)
    >>> est = SGDClassifier(tol=1e-3)
    >>> params = {'alpha': np.logspace(-4, 0, num=1000),
    >>>           'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
    >>>           'average': [True, False]}
    >>>
    >>> search = HyperbandCV(est, params)
    >>> search.fit(X, y, classes=np.unique(y))
    >>> search.best_params_
    {'loss': 'log', 'average': False, 'alpha': 0.0080502}

    Attributes
    ----------
    cv_results_ : dict of lists
        Information about the cross validation scores for each model.
        All lists are ordered the same, and this value can be imported into
        a pandas DataFrame. This dict has keys of

        * ``rank_test_score``
        * ``model_id``
        * ``mean_fit_time``
        * ``mean_score_time``
        * ``std_fit_time``
        * ``std_score_time``
        * ``mean_test_score``
        * ``std_test_score``
        * ``partial_fit_calls``
        * ``mean_train_score``
        * ``std_train_score``
        * ``params``
        * ``param_value``
        * ``mean_copy_time``

    meta_ : dict
        Information about every model that was trained. Can be used as input to
        :func:`~dask_ml.model_selection.HyperbandCV.fit_metadata`.
    history_ : list of dicts
        Information about every model after it is scored. Most models will be
        in here more than once because poor performing models are "killed" off
        early.
    best_params_ : dict
        The params that produced the best performing model
    best_estimator_ : any
        The best performing model
    best_index_ : int
        The index of the best performing classifier to be used in
        ``cv_results_``.
    n_splits_ : int
        The number of cross-validation splits.
    best_score_ : float
        The best validation score on the test set.
    best_params_ : dict
        The params that are given to the model that achieves ``best_score_``.

    Notes
    -----
    Hyperband is state of the art via an adaptive scheme. Hyperband
    only spends time on high-performing models, because our goal is to find
    the highest performing model. This means that it stops training models
    that perform poorly.

    The asynchronous variant [2]_ is controlled by the ``asynchronous``
    keyword and suited for highly parallel architectures. Architectures with
    little parallism (i.e., few workers) will benefit from
    ``asynchronous=False``.

    At least one model sees ``max_iter * batch_size`` samples, which
    should be large enough for convergence (or close to it). Often
    there are hard constraints on ``max_iter * batch_size``
    (e.g., time or deadline constraints).

    ``max_iter`` is (almost) proportional to the number of parameters
    evaluated, as well as being the maximum number of times
    ``partial_fit`` is called for any model.
    ``batch_size`` is the fraction of the dataset that each ``partial_fit``
    call sees.

    ``max_iter`` should be set to be reasonable given the problem and
    parameter search space, but ideally
    large enough so that early-stopping is beneficial. Higher values will
    evaluate more parameters. We recommend setting ``max_iter * batch_size``,
    then using :func:`~dask_ml.model_selection.HyperbandCV.fit_metadata`
    alongside natural constraints (time or deadline constraints) to determine
    ``max_iter`` and ``batch_size``.

    The authors of Hyperband use ``max_iter=300`` and ``batch_size=0.25``
    to tune deep learning models in [1]_. They tune 6 stochastic gradient
    descent parameters alongside two problem formulation hyperparameters.
    For :func:`~sklearn.linear_model.SGDClassifier`, examples of "stochastic
    gradient descent parameters" are ``learning_rate`` and ``eta0``.
    Examples of "problem formulation hyperparameters" are ``alpha`` and
    ``l1_ratio`` when ``penalty='elasticnet'``.

    There are some limitations to the `current` implementation of Hyperband:

    1. The full dataset is requested to be in distributed memory
    2. The testing dataset must fit in the memory of a single worker
    3. HyperbandCV does not implement cross validation

    References
    ----------
    .. [1] "Hyperband: A novel bandit-based approach to hyperparameter
           optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A.
           Rostamizadeh, and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    .. [2] "Massively Parallel Hyperparameter Tuning", 2018 by L. Li, K.
            Jamieson, A. Rostamizadeh, K. Gonina, M. Hardt, B. Recht, A.
            Talwalkar.  https://openreview.net/forum?id=S1Y7OOlRZ

    """

    def __init__(
        self,
        model,
        params,
        max_iter=300,
        eta=3,
        asynchronous=True,
        random_state=42,
        scoring=None,
        test_size=0.15,
    ):
        self.model = model
        self.params = params
        self.max_iter = max_iter
        self.eta = eta
        self.test_size = test_size
        self.random_state = random_state
        self.asynchronous = asynchronous

        self.best_score = None
        self.best_params = None

        super(HyperbandCV, self).__init__(model, scoring=scoring)

    def fit(self, X, y, **fit_params):
        """Find the best parameters for a particular model

        Parameters
        ----------
        X, y : array-like
        **fit_params
            Additional partial fit keyword arguments for the estimator.
        """
        N, R, brackets = _get_hyperband_params(self.max_iter, eta=self.eta)
        SHAs = [_SHA(n, r, limit=b + 1) for n, r, b in zip(N, R, brackets)]
        param_lists = [list(ParameterSampler(self.params, n)) for n in N]
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        all_info = []
        for SHA, param_list in zip(SHAs, param_lists):
            _, _, hist = _incremental_fit(
                self.model, param_list, X_train, y_train, X_test, y_test, SHA.fit
            )
            all_info += hist
        return default_client().sync(self._fit, X, y, **fit_params)

    @gen.coroutine
    def _fit(self, X, y, **fit_params):
        if isinstance(X, np.ndarray):
            X = da.from_array(X, chunks=X.shape)
        if isinstance(y, np.ndarray):
            y = da.from_array(y, chunks=y.shape)
        num_samples = sum(y.chunks[0])
        samples_per_batch = self.batch_size * num_samples
        X = X.rechunk({0: samples_per_batch})
        y = y.rechunk({0: samples_per_batch})

        # We always want a concrete scorer, so return_dask_score=False
        # We want this because we're always scoring NumPy arrays
        self.scorer_ = check_scoring(self.model, scoring=self.scoring)
        self.best_score = -np.inf
        r = yield _hyperband(
            self.model,
            self.params,
            X,
            y,
            max_iter=self.max_iter,
            fit_params=fit_params,
            eta=self.eta,
            random_state=self.random_state,
            test_size=self.test_size,
            scorer=self.scorer_,
            asynchronous=self.asynchronous,
        )
        params, model_meta_futures, history, meta = r

        self.meta_ = meta
        self.history_ = history

        cv_res, best_idx = _get_cv_results(meta, params)
        best_id = cv_res["model_id"][best_idx]

        best_model_and_meta = yield model_meta_futures[best_id]
        self.best_estimator_ = best_model_and_meta[0]

        self.cv_results_ = cv_res
        self.best_index_ = best_idx
        self.n_splits_ = 1  # TODO: increase this! It's hard-coded right now
        self.multimetric_ = False

        raise gen.Return(self)

    def fit_metadata(self, meta=None):
        """Get information about how much computation is required for
        :func:`~dask_ml.model_selection.HyperbandCV.fit`. This can be called
        before or after  :func:`~dask_ml.model_selection.HyperbandCV.fit`.

        Parameters
        ----------
        meta : dict, optional.
            Information about the computation that occured. This argument
            can be ``self.meta_``.

        Returns
        -------
        metadata : dict
            Information about the computation performed by ``fit``. Has keys

            * ``num_partial_fit_calls``, which is the total number of
              partial fit calls.
            * ``num_models``, which is the total number of models created.

        Notes
        ------
        Note that when asynchronous is True and meta is None, the amount of
        computation described by this function is a lower bound: more
        computation will be done if asynchronous is True.

        """
        if meta is None:
            bracket_info = _hyperband_paper_alg(self.max_iter, eta=self.eta)
            num_models = sum(b["num_models"] for b in bracket_info)
        else:
            brackets = toolz.groupby("bracket", meta)
            fit_call_key = "partial_fit_calls"
            bracket_info = [
                {
                    "num_models": max(vi["num_models"] for vi in v),
                    "num_" + fit_call_key: sum(vi[fit_call_key] for vi in v),
                    "bracket": k,
                    "iters": {vi[fit_call_key] for vi in v},
                }
                for k, v in brackets.items()
            ]
            num_models = len(set([r["model_id"] for r in meta]))
        for bracket in bracket_info:
            bracket["iters"] = sorted(list(bracket["iters"]))
        num_partial_fit = sum(b["num_partial_fit_calls"] for b in bracket_info)
        bracket_info = sorted(bracket_info, key=lambda x: x["bracket"])

        info = {
            "num_partial_fit_calls": num_partial_fit,
            "num_models": num_models,
            "_brackets": bracket_info,
        }
        return info


def _get_cv_results(history, params):
    scores = [h["score"] for h in history]
    best_idx = int(np.argmax(scores))
    keys = set(toolz.merge(history).keys())
    for unused in ["bracket", "iterations", "num_models", "bracket_iter", "score"]:
        keys.discard(unused)
    cv_results = {k: [h[k] for h in history] for k in keys}

    params = [params[model_id] for model_id in cv_results["model_id"]]
    cv_results["params"] = params
    params = {"param_" + k: [param[k] for param in params] for k in params[0].keys()}
    ranks = np.argsort(scores)[::-1]
    cv_results["rank_test_score"] = ranks.tolist()
    cv_results.update(params)
    return cv_results, best_idx


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper [1]_.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter
       optimization", 2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh,
       and A. Talwalkar.  https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(int(s_max + 1)))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta ** s / (s + 1)))
        r = int(R * eta ** -s)
        T = set(range(n))
        hist = {"num_models": n, "models": {n: 0 for n in range(n)}, "iters": []}
        for i in range(s + 1):
            n_i = math.floor(n * eta ** -i)
            r_i = np.round(r * eta ** i).astype(int)
            L = {model: r_i for model in T}
            hist["models"].update(L)
            hist["iters"] += [r_i]
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T) if i < to_keep}
        hists["bracket={s}".format(s=s)] = hist
    info = [
        {
            "bracket": k,
            "num_models": hist["num_models"],
            "num_partial_fit_calls": sum(hist["models"].values()),
            "iters": {int(h) for h in hist["iters"]},
        }
        for k, hist in hists.items()
    ]
    return info
