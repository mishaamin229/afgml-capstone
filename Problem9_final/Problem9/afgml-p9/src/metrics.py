"""Fairness, utility, calibration and uncertainty metrics (Problem 9).

Definitions follow the capstone spec: D=1 means 'selected for outreach'.
  s_g = P(D=1|G=g)   t_g = P(D=1|Y=1,G=g)   f_g = P(D=1|Y=0,G=g)
  dDP = range(s_g)   dEOpp = range(t_g)     dEOdds = max(range(t_g), range(f_g))
Groups with an undefined rate (zero denominator) are never silently dropped:
they are listed in `group_rates` with NaN and counted in `summary`.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def group_rates(y, d, groups, min_support: int = 30) -> pd.DataFrame:
    y, d, g = np.asarray(y), np.asarray(d), np.asarray(groups)
    rows = []
    for k in sorted(set(g.tolist())):
        m = g == k
        n, pos = int(m.sum()), int((y[m] == 1).sum()); neg = n - pos
        rows.append(dict(group=k, n=n, pos=pos, neg=neg,
            s=d[m].mean() if n else np.nan,
            tpr=d[m & (y == 1)].mean() if pos else np.nan,
            fpr=d[m & (y == 0)].mean() if neg else np.nan,
            exploratory_tpr=pos < min_support, exploratory_fpr=neg < min_support,
            exploratory_s=n < min_support))
    return pd.DataFrame(rows)


def _range(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return float(x.max() - x.min()) if x.size else np.nan


def fairness_summary(y, d, groups, min_support: int = 30, exclude_exploratory: bool = False) -> dict:
    """exclude_exploratory=False (default) keeps small groups; True gives the
    well-supported-only view. Report both when small cells matter."""
    r = group_rates(y, d, groups, min_support)
    s, t, f = r.s.copy(), r.tpr.copy(), r.fpr.copy()
    if exclude_exploratory:
        s[r.exploratory_s] = np.nan; t[r.exploratory_tpr] = np.nan; f[r.exploratory_fpr] = np.nan
    s_ok = s.dropna()
    ratio = (float(s_ok.min() / s_ok.max()) if len(s_ok) and s_ok.max() > 0 else np.nan)
    return dict(dDP=_range(s), dEOpp=_range(t),
                dEOdds=np.nanmax([_range(t), _range(f)]) if not (np.isnan(_range(t)) and np.isnan(_range(f))) else np.nan,
                sel_ratio=ratio,                       # NaN == undefined (max selection rate is 0)
                worst_group_tpr=float(np.nanmin(t)) if t.notna().any() else np.nan,
                n_groups=len(r), n_exploratory_tpr=int(r.exploratory_tpr.sum()),
                n_undefined_tpr=int(r.tpr.isna().sum()))


def utility(y, score, d=None) -> dict:
    out = dict(auroc=roc_auc_score(y, score), ap=average_precision_score(y, score))
    if d is not None:
        out["balanced_acc"] = balanced_accuracy_score(y, d)
        out["recall_at_capacity"] = float(np.asarray(d)[np.asarray(y) == 1].mean())
        out["precision_selected"] = float(np.asarray(y)[np.asarray(d) == 1].mean()) if np.sum(d) else np.nan
    return out


def decide_top_k(score, capacity: float):
    """Fixed outreach capacity: select the top `capacity` fraction (ties broken by index)."""
    score = np.asarray(score); k = int(round(capacity * len(score)))
    d = np.zeros(len(score), int)
    d[np.argsort(-score, kind="stable")[:k]] = 1
    return d


def prob_metrics(y, p, eps: float = 1e-12) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return dict(log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                brier=float(np.mean((p - y) ** 2)))


def group_calibration(y, p, groups, bins: int = 10) -> pd.DataFrame:
    """Per-group ECE (equal-width bins) and mean predicted vs observed rate."""
    y, p, g = np.asarray(y, float), np.asarray(p, float), np.asarray(groups)
    rows = []
    for k in sorted(set(g.tolist())):
        m = g == k; b = np.minimum((p[m] * bins).astype(int), bins - 1)
        ece = sum(abs(p[m][b == j].mean() - y[m][b == j].mean()) * (b == j).mean()
                  for j in range(bins) if (b == j).any())
        rows.append(dict(group=k, n=int(m.sum()), mean_pred=p[m].mean(), obs_rate=y[m].mean(), ece=float(ece)))
    return pd.DataFrame(rows)


# ------------------------------------------- computational-fairness metrics
def decision_change_by_group(d_exact, d_budget, groups) -> pd.DataFrame:
    """Per-group flip rate of decisions relative to the exact-inference reference,
    split into lost selections (1->0) and gained selections (0->1)."""
    de, db, g = map(np.asarray, (d_exact, d_budget, groups))
    rows = []
    for k in sorted(set(g.tolist())):
        m = g == k
        rows.append(dict(group=k, n=int(m.sum()), flip=float((de[m] != db[m]).mean()),
                         lost=float(((de[m] == 1) & (db[m] == 0)).mean()),
                         gained=float(((de[m] == 0) & (db[m] == 1)).mean())))
    return pd.DataFrame(rows)


def fallback_rate_gap(used_fallback, groups) -> dict:
    u, g = np.asarray(used_fallback, float), np.asarray(groups)
    rates = pd.Series(u).groupby(g).mean()
    return dict(rates=rates.to_dict(), gap=float(rates.max() - rates.min()))


def tv_distance(p, q) -> float:
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


# ----------------------------------------------------------- uncertainty
def block_indices(n: int, block_len: int, rng) -> np.ndarray:
    """Moving-block bootstrap indices (preserves date order within blocks)."""
    nb = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=nb)
    return np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]


def paired_block_bootstrap(stat_fn, n: int, block_len: int, n_boot: int = 1000,
                           seed: int = 0, alpha: float = 0.05) -> dict:
    """`stat_fn(idx)` must compute the PAIRED difference (method - baseline) on the
    same resampled rows. Returns point estimate and percentile interval.
    Describes evaluation uncertainty of fixed models, not training uncertainty."""
    rng = np.random.default_rng(seed)
    est = stat_fn(np.arange(n))
    reps = np.array([stat_fn(block_indices(n, block_len, rng)) for _ in range(n_boot)], float)
    ok = reps[~np.isnan(reps)]
    lo, hi = np.quantile(ok, [alpha / 2, 1 - alpha / 2]) if ok.size else (np.nan, np.nan)
    p = (np.nan if not ok.size else
         min(1.0, 2 * min((1 + (ok <= 0).sum()) / (ok.size + 1), (1 + (ok >= 0).sum()) / (ok.size + 1))))
    return dict(estimate=float(est), lo=float(lo), hi=float(hi), n_boot=n_boot,
                n_nan=int(np.isnan(reps).sum()), block_len=block_len, p_two_sided=float(p))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values for a declared confirmatory family."""
    p = np.asarray(pvals, float); m = len(p); order = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def supported_groups(y, gcode, n_groups: int, min_support: int = 30) -> np.ndarray:
    """Boolean mask over group codes with >= min_support positives in the FULL evaluation
    sample. Fixed once (not re-drawn per bootstrap replicate) so the bootstrap statistic is
    the same quantity as the reported `dEOpp_supported`."""
    pos = np.bincount(np.asarray(gcode)[np.asarray(y) == 1], minlength=n_groups)
    return pos >= min_support


def fast_deopp(y, d, gcode, n_groups: int, keep=None) -> float:
    """Vectorised dEOpp = range of group TPRs over groups with >=1 positive (keep=None;
    equals fairness_summary(...)['dEOpp'] without exclusion), or over the groups flagged in
    the boolean mask `keep` (use supported_groups() to reproduce dEOpp_supported).
    Used inside bootstrap loops."""
    y, d, g = np.asarray(y), np.asarray(d), np.asarray(gcode)
    pos = np.bincount(g[y == 1], minlength=n_groups).astype(float)
    tp = np.bincount(g[(y == 1) & (d == 1)], minlength=n_groups).astype(float)
    ok = pos > 0
    if keep is not None: ok = ok & np.asarray(keep, bool)
    if not ok.any(): return np.nan
    t = tp[ok] / pos[ok]
    return float(t.max() - t.min())
