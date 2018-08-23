import numpy as np
import scipy.stats
from sklearn.linear_model import SGDClassifier

import dask
import dask.array as da
from dask.distributed import Client, wait
from distributed.utils_test import loop, cluster, gen_cluster  # noqa: F401
from tornado import gen

from dask_ml.datasets import make_classification
from dask_ml.model_selection import HyperbandCV
from dask_ml.wrappers import Incremental
from dask_ml.utils import ConstantFunction
from distributed.metrics import time

import pytest


@pytest.mark.parametrize(  # noqa: F811
    "array_type,library",
    [("dask.array", "dask-ml"), ("numpy", "sklearn"), ("numpy", "test")],
)
def test_sklearn(array_type, library, loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            n, d = (200, 2)

            rng = da.random.RandomState(42)

            # create observations we know linear models can fit
            X = rng.normal(size=(n, d), chunks=n // 2)
            coef_star = rng.uniform(size=d, chunks=d)
            y = da.sign(X.dot(coef_star))

            if array_type == "numpy":
                X = X.compute()
                y = y.compute()

            params = {
                "loss": [
                    "hinge",
                    "log",
                    "modified_huber",
                    "squared_hinge",
                    "perceptron",
                ],
                "average": [True, False],
                "learning_rate": ["constant", "invscaling", "optimal"],
            }
            model = SGDClassifier(
                tol=-np.inf, penalty="elasticnet", random_state=42, eta0=0.1
            )
            if library == "dask-ml":
                model = Incremental(model)
                params = {"estimator__" + k: v for k, v in params.items()}
            elif library == "test":
                model = ConstantFunction()
                params = {"value": np.linspace(0, 1, num=1000)}

            search = HyperbandCV(
                model, params, max_iter=27, random_state=42, asynchronous=False
            )
            search.fit(X, y, classes=da.unique(y))

            score = search.best_estimator_.score(X, y)
            if library == "sklearn":
                assert score > 0.67
            if library == "dask-ml":
                assert score > 0.67
            elif library == "test":
                assert score > 0.9
            assert type(search.best_estimator_) == type(model)
            assert isinstance(search.best_params_, dict)

            num_fit_models = len(set(search.cv_results_["model_id"]))
            assert num_fit_models == 49
            best_idx = search.best_index_
            assert search.cv_results_["test_score"][best_idx] == max(
                search.cv_results_["test_score"]
            )


@pytest.mark.parametrize("library", ["sklearn", "dask-ml"])  # noqa: F811
def test_scoring_param(loop, library):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            assert library in {"dask-ml", "sklearn"}
            max_iter = 9
            X, y = make_classification(
                n_samples=100, n_features=20, random_state=42, chunks=20
            )
            X, y = dask.persist(X, y)
            if library == "sklearn":
                X = X.compute()
                y = y.compute()

            kwargs = dict(tol=1e-3, penalty="elasticnet", random_state=42)

            model = SGDClassifier(**kwargs)
            params = {
                "alpha": np.logspace(-2, 1, num=1000),
                "l1_ratio": np.linspace(0, 1, num=1000),
                "average": [True, False],
            }
            if library == "dask-ml":
                model = Incremental(model)
                params = {'estimator__' + k: v for k, v in params.items()}

            alg1 = HyperbandCV(
                model, params, max_iter=max_iter, scoring="accuracy",
                random_state=42
            )
            alg1.fit(X, y, classes=da.unique(y))

            alg2 = HyperbandCV(
                model, params, max_iter=max_iter, scoring="r2", random_state=42
            )
            alg2.fit(X, y, classes=da.unique(y))

            assert alg1.scoring != alg2.scoring
            assert alg1.scorer_ != alg2.scorer_
            assert alg1.score(X, y) != alg2.score(X, y)


@pytest.mark.skip(reason="asynchronous not implemented")
def test_async_keyword(loop):  # noqa: F811
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            max_iter = 27
            X, y = make_classification(chunks=20)
            model = ConstantFunction()

            params = {"value": np.logspace(-2, 1, num=max_iter)}
            alg0 = HyperbandCV(
                model, params, asynchronous=False, max_iter=max_iter,
                random_state=42
            )
            alg0.fit(X, y)

            alg1 = HyperbandCV(
                model, params, asynchronous=True, max_iter=max_iter,
                random_state=42
            )
            alg1.fit(X, y)

            info0 = alg0.fit_metadata(meta=alg0.meta_)
            info1 = alg1.fit_metadata(meta=alg1.meta_)
            assert info0["num_models"] == info1["num_models"]
            assert alg0.score(X, y) == alg1.score(X, y)


@pytest.mark.skip(reason="asynchronous not implemented")
@gen_cluster(client=True)
def test_sklearn_async(c, s, a, b):
    max_iter = 27
    chunk_size = 20
    X, y = make_classification(
        n_samples=100, n_features=20, random_state=42, chunks=chunk_size
    )
    X, y = dask.persist(X, y)
    yield wait([X, y])

    kwargs = dict(tol=1e-3, penalty="elasticnet", random_state=42)

    model = SGDClassifier(**kwargs)

    params = {
        "alpha": np.logspace(-2, 1, num=1000),
        "l1_ratio": np.linspace(0, 1, num=1000),
        "average": [True, False],
    }
    search = HyperbandCV(model, params, max_iter=max_iter, random_state=42)
    s_tasks = set(s.tasks)
    c_futures = set(c.futures)
    yield search._fit(X, y, classes=da.unique(y))

    assert set(c.futures) == c_futures
    start = time()
    while set(s.tasks) != s_tasks:
        yield gen.sleep(0.01)
        assert time() < start + 5

    assert len(set(search.cv_results_["model_id"])) == 49


@pytest.mark.parametrize("max_iter", [27])#, 81])  # noqa: F811
def test_hyperband_mirrors_paper(loop, max_iter):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):

            X, y = make_classification(chunks=5, n_features=5)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(
                model, params, max_iter=max_iter, random_state=0,
                asynchronous=False
            )
            alg.fit(X, y)

            paper_info = alg.fit_metadata()
            actual_info = alg.meta_
            assert paper_info == actual_info


def test_integration(loop):  # noqa: F811
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop):
            X, y = make_classification(n_samples=10, n_features=4, chunks=10)
            model = ConstantFunction()
            params = {"value": scipy.stats.uniform(0, 1)}
            alg = HyperbandCV(
                model, params, asynchronous=True, max_iter=9, random_state=42
            )
            alg.fit(X, y)
            cv_res_keys = set(alg.cv_results_.keys())
            gt_zero = lambda x: x >= 0
            for column, dtype, condition in [
                ("params", dict, lambda d: set(d.keys()) == {"value"}),
                ("test_score", float, None),
                ("mean_test_score", float, None),
                ("rank_test_score", np.int64, None),
                ("mean_partial_fit_time", float, gt_zero),
                ("std_partial_fit_time", float, gt_zero),
                ("mean_score_time", float, gt_zero),
                ("std_score_time", float, gt_zero),
                ("model_id", str, None),
                ("partial_fit_calls", int, gt_zero),
                ("param_value", float, None),
            ]:
                if dtype:
                    assert all(isinstance(x, dtype)
                               for x in alg.cv_results_[column])
                if condition:
                    assert all(condition(x) for x in alg.cv_results_[column])
                cv_res_keys -= {column}

            # the keys listed in the for-loop are all the keys in cv_results_
            assert cv_res_keys == set()

            alg.best_estimator_.fit(X, y)
            alg.best_estimator_.score(X, y)
            alg.fit(X, y)
            alg.score(X, y)
            assert isinstance(alg.best_index_, int)
            assert isinstance(alg.best_score_, float)
            assert isinstance(alg.best_estimator_, ConstantFunction)
            assert isinstance(alg.best_params_, dict)
