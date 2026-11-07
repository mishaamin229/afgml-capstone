"""Smoothed discrete BN fitted on TRAIN only; per-row exact/budgeted prediction."""
from __future__ import annotations
import numpy as np, pandas as pd
from src.inference import (Factor, variable_elimination, budgeted_query, prune_barren,
                           BudgetExceeded, CliqueTree, brute_force, OpCounter)
from src.data import assert_no_leakage, MASKABLE


def fit_bn(train: pd.DataFrame, structure: dict, alpha: float = 1.0, cfg: dict | None = None):
    """Return (factors, states). States come from the training split; categories
    unseen at test time are later treated as unobserved (never added post hoc)."""
    if cfg is not None:
        assert_no_leakage([v for v in structure if v != "y"], cfg)
    states = {v: ([0, 1] if v == "y" else sorted(train[v].astype(str).unique())) for v in structure}
    idx = lambda v: train[v].map({s: i for i, s in enumerate(states[v])}).to_numpy()
    factors = []
    for child, pa in structure.items():
        vs = list(pa) + [child]
        shape = [len(states[v]) for v in vs]
        counts = np.full(shape, float(alpha))
        np.add.at(counts, tuple(idx(v) if v != "y" else train["y"].to_numpy() for v in vs), 1.0)
        factors.append(Factor(tuple(vs), counts / counts.sum(axis=-1, keepdims=True)))
    return factors, states


def row_evidence(row, states, unknown_as_masked=True) -> dict:
    ev = {}
    for v, ss in states.items():
        if v == "y": continue
        val = str(row[v])
        if unknown_as_masked and val == "unknown" and v in MASKABLE: continue
        if val in ss: ev[v] = ss.index(val)           # unseen category -> unobserved
    return ev


def evidence_matrix(df, states, unknown_as_masked=True):
    """Integer evidence matrix (rows x non-target vars); -1 = unobserved.
    'unknown' (maskable columns) and categories unseen in training are unobserved."""
    vs = [v for v in states if v != "y"]
    E = np.full((len(df), len(vs)), -1, dtype=int)
    for j, v in enumerate(vs):
        col = df[v].astype(str)
        codes = col.map({s_: i for i, s_ in enumerate(states[v])}).to_numpy(dtype=float)
        codes = np.where(np.isnan(codes), -1, codes).astype(int)
        if unknown_as_masked and v in MASKABLE:
            codes = np.where(col.to_numpy() == "unknown", -1, codes)
        E[:, j] = codes
    return vs, E


def predict(df: pd.DataFrame, factors, states, budget=None, heuristic="min_fill",
            simplify=True, fallback=None, extra_mask: pd.DataFrame | None = None,
            unknown_as_masked=True) -> pd.DataFrame:
    """P(y=1|evidence) per row under an optional deterministic op budget.
    `extra_mask` (bool DataFrame, True = hide that variable) injects SYNTHETIC
    unobserved variables. Rows with identical evidence share one computation."""
    vs, E = evidence_matrix(df, states, unknown_as_masked)
    if extra_mask is not None:
        for j, v in enumerate(vs):
            if v in extra_mask.columns:
                E[extra_mask[v].to_numpy(bool), j] = -1
    uniq, inv = np.unique(E, axis=0, return_inverse=True)
    res = []
    for row in uniq:
        ev = {v: int(c) for v, c in zip(vs, row) if c >= 0}
        r = budgeted_query(factors, ["y"], ev, budget, heuristic, simplify, fallback)
        res.append((float(r["posterior"][1]), r["ops"], r["peak"], r["used_fallback"], len(ev)))
    out = pd.DataFrame(res, columns=["p1", "ops", "peak", "used_fallback", "n_evidence"]).iloc[inv.ravel()]
    out.index = df.index
    return out


def verify_cgm_fitted(factors, states, df, n_patterns=25, tol=1e-6, seed=0) -> dict:
    """C-GM on the FITTED model: VE (both heuristics) vs clique tree vs enumeration
    over real and synthetic evidence patterns. Returns max absolute errors."""
    rng = np.random.default_rng(seed)
    cts = {h: CliqueTree(factors, h) for h in ("min_fill", "min_weight")}
    rows = df.sample(min(n_patterns, len(df)), random_state=seed)
    worst = dict(ve_vs_ct=0.0, ve_vs_enum=0.0, ct_vs_enum=0.0, p_evidence=0.0)
    for _, row in rows.iterrows():
        ev = row_evidence(row, states)
        drop = [v for v in ev if rng.random() < 0.3]         # synthetic extra masking
        ev = {v: s for v, s in ev.items() if v not in drop}
        ref, zref = brute_force(factors, ["y"], ev)
        for h, ct in cts.items():
            p, z = variable_elimination(factors, ["y"], ev, h)
            ct.calibrate(ev); pc, zc = ct.marginal(["y"])
            worst["ve_vs_ct"] = max(worst["ve_vs_ct"], float(np.abs(p - pc).max()))
            worst["ve_vs_enum"] = max(worst["ve_vs_enum"], float(np.abs(p - ref).max()))
            worst["ct_vs_enum"] = max(worst["ct_vs_enum"], float(np.abs(pc - ref).max()))
            worst["p_evidence"] = max(worst["p_evidence"], abs(z - zref), abs(zc - zref))
    worst.update(tolerance=tol, passed=bool(max(v for k, v in worst.items() if k != "tolerance") <= tol),
                 n_patterns=len(rows))
    return worst
