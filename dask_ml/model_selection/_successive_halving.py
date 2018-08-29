import numpy as np
import math
import toolz
from time import time


def _get_hist(info):
    hist = [v[-1] for v in info.values()]
    for h in hist:
        h['wall_time'] = time()
    return hist


def stop_on_plateau(info, patience=10, tol=0.001, max_iter=None):
    """
    Stop training when a plateau is reaching with validation score.

    A plateau is defined to be when the validation scores are all below the
    plateau's start score, plus the tolerance. The plateau is ``patience``
    ``partial_fit`` calls wide. That is, a plateau if reached if for every
    score,

        score < plateau_start_score + tol

    This function is designed for use with
    :func:`~dask_ml.model_selection.fit`.

    Parameters
    ----------
    patience : int
        Number of partial_fit_calls that specifies the plateau's width
    tol : float
        The tolerance that specifies the plateau.
    max_iter : int
        How many times to call ``partial_fit`` on each model.

    Returns
    -------
    partial_fit_calls : dict
        Each key specifies wether to continue training this model.
    """
    out = {}
    for ident, records in info.items():
        pf_calls = records[-1]['partial_fit_calls']
        if max_iter is not None and pf_calls >= max_iter:
            out[ident] = 0

        elif pf_calls >= patience:
            plateau = {d['partial_fit_calls']: d['score']
                       for d in records
                       if d['partial_fit_calls'] >= pf_calls - patience}
            plateau_start = plateau[min(plateau)]
            if all(score < plateau_start + tol for score in plateau.values()):
                out[ident] = 0
            else:
                out[ident] = 1
        else:
            out[ident] = 1
    return out


class _HistoryRecorder:
    def __init__(self, fn, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.history = []

    def fit(self, info):
        self.history += _get_hist(info)
        return self.fn(info, *self.args, **self.kwargs)


class _SHA:
    def __init__(self, n, r, eta=3, limit=None, patience=10, tol=0.001):
        """
        Perform the successive halving algorithm.

        Parameters
        ----------
        n : int
            Number of models to evaluate initially
        r : int
            Number of times to call partial fit initially
        eta : float, default=3
            How aggressive to be in culling off the models. Higher
            values correspond to being more aggressive in killing off
            models. The "infinite horizon" theory suggests eta=np.e=2.718...
            is optimal.
        patience : int
            Passed to `stop_on_plateau`
        tol : float
            Passed to `stop_on_plateau`
        """
        self.steps = 0
        self.n = n
        self.r = r
        self.eta = eta
        self.meta = []
        self.limit = limit
        self._best_scores = {}
        self._history = []
        self.patience = patience
        self.tol = tol

    def fit(self, info):
        self._history += _get_hist(info)
        for ident, hist in info.items():
            self._best_scores[ident] = hist[-1]['score']
        n, r, eta = self.n, self.r, self.eta
        n_i = int(math.floor(n * eta ** -self.steps))
        r_i = np.round(r * eta**self.steps).astype(int)

        # Initial case
        # partial fit has already been called once
        if r_i == 1:
            # if r_i == 1, a step has already been completed for us
            assert self.steps == 0
            self.steps = 1
            pf_calls = {k: info[k][-1]['partial_fit_calls'] for k in info}
            return self.fit(info)
        # this ordering is important; typically r_i==1 only when steps==0
        if self.steps == 0:
            # we have r_i - 1 more steps to train to
            self.steps = 1
            return {k: r_i - 1 for k in info}

        keep_training = stop_on_plateau(info,
                                        patience=self.patience,
                                        tol=self.tol)
        if sum(keep_training.values()) == 0:
            return {k: 0 for k in self._best_scores}
        info = {k: info[k] for k in keep_training}

        best = toolz.topk(n_i, info, key=lambda k: info[k][-1]['score'])
        self.steps += 1

        if self.steps > self.limit or (self.limit is None and len(best) in {0, 1}):
            max_score = max(self._best_scores.values())
            best_ids = {k for k, v in self._best_scores.items() if v == max_score}
            return {best_id: 0 for best_id in best_ids}

        pf_calls = {k: info[k][-1]['partial_fit_calls'] for k in best}
        addtl_pf_calls = {k: r_i - pf_calls[k]
                          for k in best}
        dont_train = {k: 0 for k in self._best_scores if k not in addtl_pf_calls}
        assert set(addtl_pf_calls).intersection(dont_train) == set()
        addtl_pf_calls.update(dont_train)
        return addtl_pf_calls
