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
            best_id = max(info, key=lambda k: info[k][-1]['score'])
            return {best_id: 0}

        pf_calls = {k: info[k][-1]['partial_fit_calls'] for k in best}
        addtl_pf_calls = {k: r_i - pf_calls[k]
                          for k in best}
        return addtl_pf_calls
