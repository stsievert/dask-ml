import numpy as np
import math
import toolz


def stop_on_plateau(info, patience=10, tol=0.001, max_iter=None):
    out = {}
    for ident, records in info.items():
        pf_calls = records[-1]['partial_fit_calls']
        if max_iter is not None and pf_calls > max_iter:
            out[ident] = 0

        elif pf_calls > patience:
            # old = records[-patience]['score']
            plateau = {d['partial_fit_calls']: d['score']
                       for d in records
                       if pf_calls - patience <= d['partial_fit_calls']}
            plateau_start = plateau[min(plateau)]
            if all(score < plateau_start + tol for score in plateau.values()):
                out[ident] = 0
            else:
                out[ident] = 1
        else:
            out[ident] = 1
    return out


class _SHA:
    def __init__(self, n, r, eta=3, limit=None,
                 patience=np.inf, tol=0.001):
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
        tol : int
            Passed to `stop_on_plateau`
        """
        self.steps = 0
        self.n = n
        self.r = r
        self.eta = eta
        self.meta = []
        self.patience = patience
        self.tol = tol
        self.limit = limit

    def fit(self, info):
        n, r, eta = self.n, self.r, self.eta
        n_i = math.floor(n * eta ** -self.steps)
        r_i = np.round(r * eta**self.steps).astype(int)

        # Initial case
        # partial fit has already been called once
        if r_i == 1:
            # if r_i == 1, a step has already been completed for us
            assert self.steps == 0
            self.steps = 1
            pf_calls = {k: info[k][-1]['partial_fit_calls'] for k in info}
            return self.fit(info)
        # this ordering is important; typically r_i==1 when steps==0
        if self.steps == 0:
            # we have r_i - 1 more steps to train to
            self.steps = 1
            return {k: r_i - 1 for k in info}

        keep_training = stop_on_plateau(info,
                                        patience=self.patience,
                                        tol=self.tol)
        if sum(keep_training.values()) == 0:
            return keep_training
        info = {k: info[k] for k in keep_training}

        best = toolz.topk(n_i, info, key=lambda k: info[k][-1]['score'])
        self.steps += 1

        if len(best) in {0, 1} and self.steps > self.limit:
            return {0: 0}

        pf_calls = {k: info[k][-1]['partial_fit_calls'] for k in best}
        addtl_pf_calls = {k: r_i - pf_calls[k]
                          for k in best}
        return addtl_pf_calls

from sklearn.model_selection import ParameterSampler
from sklearn.base import clone
from dask_ml.model_selection._incremental import fit
from dask_ml.datasets import make_classification
from distributed import Client
from _hyperband import _hyperband_paper_alg
from dask_ml.utils import ConstantFunction

if __name__ == "__main__":
    client = Client('localhost:8786')
    X, y = make_classification(n_features=5, n_samples=200, chunks=10)
    R = 100
    eta = 3.0
    # def hyperband(R, eta=3):
    info = _hyperband_paper_alg(R, eta=eta)
    # Because we call `partial_fit` before
    for i in info:
        i['iters'].update({1})

    sh_info = []
    s_max = math.floor(math.log(R, eta))
    B = (s_max + 1) * R
    for s in reversed(np.arange(s_max + 1)):
        n = np.ceil(B / R * eta**s / (s + 1))
        r = np.floor(R * eta**-s)
        alg = _SHA(n, r, limit=s + 1)
        model = ConstantFunction()
        params = {'value': np.linspace(0, 1, num=1000)}
        params_list = list(ParameterSampler(params, n))

        _, _, hist = fit(model, params_list, X, y, X, y, alg.fit)
        ids = {h['model_id'] for h in hist}
        info_hist = {i: [] for i in ids}
        for h in hist:
            info_hist[h['model_id']] += [h]
        hist = info_hist

        calls = {k: max(hi['partial_fit_calls'] for hi in h)
                 for k, h in hist.items()}
        iters = {hi['partial_fit_calls'] for h in hist.values() for hi in h}
        sh_info += [{'bracket': f'bracket={s}',
                     'iters': iters,
                     'num_models': len(hist),
                     'num_partial_fit_calls': sum(calls.values())}]

    assert sh_info == info
