import pytest
import numpy as np
import scipy.stats as stats
import math

from sklearn.linear_model import SGDClassifier

from distributed import Client
from distributed.utils_test import loop, cluster, gen_cluster
import dask.array as da

from dask_ml.datasets import make_classification
from dask_ml.model_selection import HyperbandCV
from dask_ml.wrappers import Incremental
from dask_ml.model_selection._hyperband import _top_k
from dask_ml.utils import ConstantFunction
import itertools


def test_top_k():
    in_ = [{'score': 0, 'model': '0'},
           {'score': 1, 'model': '1'},
           {'score': 2, 'model': '2'},
           {'score': 3, 'model': '3'},
           {'score': 4, 'model': '4'}]
    out = _top_k(in_, k=2, sort="score")
    assert out == [{'score': 4, 'model': '4'},
                   {'score': 3, 'model': '3'}]


@pytest.mark.parametrize("array_type,library", [("dask.array", "dask-ml"),
                                                ("numpy", "sklearn"),
                                                ("numpy", "test")])
def test_sklearn(array_type, library, loop, max_iter=27):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop):
            chunk_size = 100  # make dask array with one chunk
            X, y = make_classification(n_samples=chunk_size, n_features=20,
                                       random_state=42, chunks=chunk_size)
            if array_type == "numpy":
                X = X.compute()
                y = y.compute()
                chunk_size = X.shape[0]

            kwargs = dict(tol=1e-3, penalty='elasticnet', random_state=42)
            models = {'sklearn': SGDClassifier(**kwargs),
                      'dask-ml': Incremental(SGDClassifier(), **kwargs),
                      'test': ConstantFunction()}
            sgd_params = {'alpha': np.logspace(-2, 1, num=1000),
                          'l1_ratio': np.linspace(0, 1, num=1000),
                          'average': [True, False]}
            all_params = {'sklearn': sgd_params, 'dask-ml': sgd_params,
                          'test': {'value': np.linspace(0, 1)}}
            model = models[library]
            params = all_params[library]

            search = HyperbandCV(model, params, max_iter=max_iter,
                                 random_state=42, asynchronous=False)
            search.fit(X, y, classes=da.unique(y))

            models = {k: v[0] for k, v in search._models_and_meta.items()}
            trained = [hasattr(model, "t_") for model in models.values()]
            assert all(trained)

            def _iters(model):
                t_ = (model.estimator.t_ if hasattr(model, 'estimator')
                      else model.t_)
                # Test fraction of 0.15 is hardcoded into _hyperband
                return (t_ - 1) / (chunk_size * (1 - 0.15))
            iters = {_iters(model) for model in models.values()}

            info_plain = search.info()
            info_train = search.info(jobs=search.jobs_)
            assert info_plain['brackets'] == info_train['brackets']
            assert info_train == info_plain
            assert 1 <= min(iters) < max(iters) <= max_iter


@gen_cluster(client=True)
async def test_sklearn_async(c, s, a, b):
    max_iter = 27
    chunk_size = 20
    X, y = make_classification(n_samples=100, n_features=20,
                               random_state=42, chunks=chunk_size)

    kwargs = dict(tol=1e-3, penalty='elasticnet', random_state=42)

    model = SGDClassifier(**kwargs)

    params = {'alpha': np.logspace(-2, 1, num=1000),
              'l1_ratio': np.linspace(0, 1, num=1000),
              'average': [True, False]}
    search = HyperbandCV(model, params, max_iter=max_iter,
                         random_state=42, asynchronous=False)
    await search._fit(X, y, classes=da.unique(y))

    models = [v[0] for v in (await search._models_and_meta).values()]
    trained = [hasattr(model, "coef_") for model in models]
    print("__58", sum(trained) / len(trained), sum(trained), len(trained))
    assert all(trained)

    def _iters(model):
        t_ = (model.estimator.t_ if hasattr(model, 'estimator')
              else model.t_)
        # Test fraction of 0.15 is hardcoded into _hyperband
        return (t_ - 1) / (chunk_size * (1 - 0.15))
    iters = {_iters(model) for model in models}
    assert len(iters) > 1
    assert 1 <= min(iters) < max(iters) <= max_iter


@pytest.mark.parametrize("max_iter", [3, 9, 27, 81])
def test_info(loop, max_iter):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            model = ConstantFunction()
            params = {'value': stats.uniform(0, 1)}
            alg = HyperbandCV(model, params, max_iter=max_iter, random_state=0,
                              asynchronous=False)
            info = alg.info()
            paper_alg_info = _hyperband_paper_alg(max_iter)
            saved = {81: {'bracket=4': 243, 'bracket=3': 222, 'bracket=2': 225,
                          'bracket=1': 324, 'bracket=0': 405},
                     27: {'bracket=3': 63, 'bracket=2': 60, 'bracket=1': 90,
                          'bracket=0': 108},
                     9: {'bracket=2': 15, 'bracket=1': 15, 'bracket=0': 27},
                     3: {'bracket=1': 3, 'bracket=0': 6}}
            print(info)
            assert paper_alg_info == saved[max_iter]
            assert (info['total_partial_fit_calls'] ==
                    sum(paper_alg_info.values()) )#==
                    #  sum(b['partial_fit_calls'] for b in info['brackets']))
            for bracket in info['brackets']:
                k = bracket['bracket']
                assert bracket['partial_fit_calls'] == paper_alg_info[k]
            assert (info['total_models'] ==
                    sum(b['num_models'] for b in info['brackets']))

            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            alg.fit(X, y)
            info_after_fit = alg.info(jobs=alg.jobs_)
            assert info_after_fit == info


def test_integration(loop):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            model = ConstantFunction()
            params = {'value': stats.uniform(0, 1)}
            alg = HyperbandCV(model, params)
            alg.fit(X, y)
            cv_res_keys = set(alg.cv_results_.keys())
            assert cv_res_keys == {'rank_test_score', "model_id",
                                   'mean_fit_time', 'mean_score_time',
                                   'std_fit_time', 'std_score_time',
                                   'mean_test_score', 'std_test_score',
                                   'partial_fit_calls', 'mean_train_score',
                                   'std_train_score', 'params', 'param_value',
                                   'mean_copy_time'}
            for column, dtype in [('rank_test_score', int),
                                  ('model_id', str),
                                  ('mean_score_time', float),
                                  ('mean_test_score', float),
                                  ('mean_fit_time', float),
                                  ('partial_fit_calls', int),
                                  ('params', dict),
                                  ('param_value', float),
                                  ('mean_copy_time', float)]:
                assert all(isinstance(v, dtype)
                           for v in alg.cv_results_[column])
            alg.best_estimator_.fit(X, y)
            assert isinstance(alg.best_index_, int)
            assert isinstance(alg.best_score_, float)
            assert isinstance(alg.best_estimator_, ConstantFunction)
            assert isinstance(alg.best_params_, dict)


def test_copy(loop):
    """
    Make sure models are not changed in place, and have the same parameters
    each time
    """
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            rng = np.random.RandomState(42)
            n, d = 100, 1
            X = (np.arange(n * d) // d).reshape(n, d)
            y = np.sign(rng.randn(n))
            X = da.from_array(X, (2, d))
            y = da.from_array(y, 2)

            model = ConstantFunction()
            params = {'value': np.logspace(-2, 1, num=1000)}

            max_iter = 27
            alg = HyperbandCV(model, params, max_iter=max_iter,
                              random_state=42)
            alg.fit(X, y, classes=da.unique(y))

            all_models = alg._all_models

    copied_models = list(filter(lambda x: len(x) > 1,
                              all_models.values()))
    assert len(copied_models) > 1
    for models in copied_models:
        assert all(not np.allclose(m1.coef_, m2.coef_)
                   for m1, m2 in itertools.combinations(models, 2))


def _hyperband_paper_alg(R, eta=3):
    """
    Algorithm 1 from the Hyperband paper. Only a slight modification is made,
    the ``if to_keep <= 1``: if 1 model is left there's no sense in training
    any further.

    References
    ----------
    1. "Hyperband: A novel bandit-based approach to hyperparameter optimization",
       2016 by L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar.
       https://arxiv.org/abs/1603.06560
    """
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    brackets = reversed(range(int(s_max + 1)))
    hists = {}
    for s in brackets:
        n = int(math.ceil(B / R * eta**s / (s + 1)))
        r = int(R * eta**-s)

        T = {n for n in range(n)}
        hist = {n: 0 for n in range(n)}
        for i in range(s + 1):
            n_i = math.floor(n * eta**-i)
            r_i = r * eta**i
            L = {model: r_i for model in T}
            hist.update(L)
            to_keep = math.floor(n_i / eta)
            T = {model for i, model in enumerate(T)
                 if i < to_keep}
            if to_keep <= 1:
                break

        hists['bracket={s}'.format(s=s)] = hist

    num_partial_fit_calls = {k: sum(v.values()) for k, v in hists.items()}
    return num_partial_fit_calls