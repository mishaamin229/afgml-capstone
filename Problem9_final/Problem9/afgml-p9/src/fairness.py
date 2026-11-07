"""Equal-opportunity threshold adjustment (Hardt et al., 2016 style post-processing),
matched to a target selection capacity. Fitted on VALIDATION data only."""
from __future__ import annotations
import numpy as np


def fit_eopp_thresholds(score, y, groups, capacity, min_pos=30, iters=40):
    score, y, g = np.asarray(score, float), np.asarray(y), np.asarray(groups)
    thr_global = float(np.quantile(score, 1 - capacity))
    labels = sorted(set(g.tolist()))
    sc_pos = {k: np.sort(score[(g == k) & (y == 1)])[::-1] for k in labels}
    ok = [k for k in labels if len(sc_pos[k]) >= min_pos]

    def thresholds(t):
        out = {}
        for k in labels:
            if k in ok:
                kk = max(1, int(np.ceil(t * len(sc_pos[k]))))
                out[k] = float(sc_pos[k][kk - 1])        # select score >= threshold
            else:
                out[k] = thr_global                      # too few positives: keep global rule
        return out

    def sel_rate(t):
        th = thresholds(t)
        return float(np.mean(score >= np.array([th[k] for k in g])))

    lo, hi = 0.0, 1.0
    for _ in range(iters):                                # sel_rate is nondecreasing in t
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sel_rate(mid) < capacity else (lo, mid)
    return thresholds((lo + hi) / 2), thr_global


def apply_thresholds(score, groups, thr, default):
    th = np.array([thr.get(k, default) for k in np.asarray(groups)])
    return (np.asarray(score, float) >= th).astype(int)
