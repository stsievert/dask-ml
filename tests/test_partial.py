from sklearn.linear_model import SGDClassifier
from sklearn.base import clone
import numpy as np
import dask
import dask.array as da
from dask_ml._partial import fit, predict
from dask_ml.datasets import make_classification
from dask_ml.wrappers import Incremental
import pytest


x = np.array([[1, 0],
              [2, 0],
              [3, 0],
              [4, 0],
              [0, 1],
              [0, 2],
              [3, 3],
              [4, 4]])

y = np.array([1, 1, 1, 1, -1, -1, 0, 0])

z = np.array([[1, -1],
              [-1, 1],
              [10, -10],
              [-10, 10]])

X = da.from_array(x, chunks=(3, 2))
Y = da.from_array(y, chunks=(3,))
Z = da.from_array(z, chunks=(2, 2))


def test_fit():
    with dask.config.set(scheduler='single-threaded'):
        sgd = SGDClassifier(max_iter=5)

        sgd = fit(sgd, X, Y, classes=np.array([-1, 0, 1]))

        sol = sgd.predict(z)
        result = predict(sgd, Z)
        assert result.chunks == ((2, 2),)
        assert result.compute().tolist() == sol.tolist()


def test_fit_rechunking():
    n_classes = 2
    X, y = make_classification(chunks=20, n_classes=n_classes)
    X = X.rechunk({1: 10})

    assert X.numblocks[1] > 1

    clf = Incremental(SGDClassifier(max_iter=5))
    clf.fit(X, y, classes=list(range(n_classes)))


def test_fit_shuffle_blocks():
    X, y = make_classification(random_state=0, chunks=20)
    sgd = SGDClassifier(max_iter=5)

    fit(clone(sgd), X, y, classes=np.array([-1, 0, 1]),
        shuffle_blocks=False)
    fit(clone(sgd), X, y, classes=np.array([-1, 0, 1]),
        shuffle_blocks=True)

    fit(clone(sgd), X, y, classes=np.array([-1, 0, 1]),
        shuffle_blocks=True, random_state=42)
    fit(clone(sgd), X, y, classes=np.array([-1, 0, 1]),
        shuffle_blocks=True, random_state=np.random.RandomState(42))

    with pytest.raises(ValueError, match='cannot be used to seed'):
        fit(sgd, X, y, classes=np.array([-1, 0, 1]),
            shuffle_blocks=True, random_state=da.random.RandomState(42))
