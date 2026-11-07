"""Experiments 1-2: exact-inference agreement (C-GM) and cost/memory profiling.
All costs are DETERMINISTIC operation counts; wall-clock lives in scripts/profile_latency.py."""
from __future__ import annotations
import tracemalloc
import numpy as np, pandas as pd
from src.bn import evidence_matrix
from src.inference import (variable_elimination, CliqueTree, brute_force, OpCounter)

HEURISTICS = ("min_fill", "min_weight", "min_degree")


def _patterns(df, states, mask=None, cap=300, seed=0):
    vs, E = evidence_matrix(df, states)
    if mask is not None:
        for j, v in enumerate(vs):
            if v in mask.columns: E[mask[v].to_numpy(bool), j] = -1
    U = np.unique(E, axis=0)
    if len(U) > cap: U = U[np.random.default_rng(seed).choice(len(U), cap, replace=False)]
    return [{v: int(c) for v, c in zip(vs, r) if c >= 0} for r in U]


def exact_agreement(factors, states, df, mask=None, n_patterns=300, n_enum=30, tol=1e-6):
    pats = _patterns(df, states, mask, n_patterns)
    cts = {h: CliqueTree(factors, h) for h in ("min_fill", "min_weight")}
    w = dict(order=0.0, ve_ct=0.0, enum=0.0, pe=0.0)
    for i, ev in enumerate(pats):
        post = {h: variable_elimination(factors, ["y"], ev, h) for h in HEURISTICS}
        ref_p, ref_z = post["min_fill"]
        for h in HEURISTICS[1:]:
            w["order"] = max(w["order"], float(np.abs(post[h][0] - ref_p).max()), abs(post[h][1] - ref_z))
        for h, ct in cts.items():
            ct.calibrate(ev); pc, zc = ct.marginal(["y"])
            w["ve_ct"] = max(w["ve_ct"], float(np.abs(pc - ref_p).max()))
            w["pe"] = max(w["pe"], abs(zc - ref_z))
        if i < n_enum:
            e, ez = brute_force(factors, ["y"], ev)
            w["enum"] = max(w["enum"], float(np.abs(e - ref_p).max()), abs(ez - ref_z))
    mx = max(w.values())
    return dict(max_err_across_orders=w["order"], max_err_ve_vs_clique_tree=w["ve_ct"],
                max_err_vs_enumeration=w["enum"], max_err_p_evidence=w["pe"], tolerance=tol,
                passed=bool(mx <= tol), n_patterns=len(pats), n_enumeration=min(n_enum, len(pats)))


def complexity_profile(factors, states, df, mask_vars, mask_fracs=(0.25, 0.5, 0.75), cap=200, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    wl = {"natural": None}
    for f in mask_fracs:
        wl[f"synthetic_mask_{int(f*100)}"] = pd.DataFrame(
            {v: rng.random(len(df)) < f for v in mask_vars}, index=df.index)
    for wname, mask in wl.items():
        pats = _patterns(df, states, mask, cap)
        for h in HEURISTICS:
            ops, pk = [], []
            for ev in pats:
                c = OpCounter(); variable_elimination(factors, ["y"], ev, h, c)
                ops.append(c.total); pk.append(c.peak_factor_size)
            tracemalloc.start(); variable_elimination(factors, ["y"], pats[0], h); mem = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            rows.append(dict(workload=wname, method=f"ve_{h}", n_patterns=len(pats), ops_mean=np.mean(ops),
                             ops_max=max(ops), peak_factor_mean=np.mean(pk), peak_factor_max=max(pk),
                             mem_peak_kb=mem / 1024, preprocessing_ops=0, max_clique=np.nan, n_cliques=np.nan))
        for h in ("min_fill", "min_weight"):
            ct = CliqueTree(factors, h)
            c0 = ct.calibrate({}); pre_ops, pre_peak = c0.total, c0.peak_factor_size
            ops, pk = [], []
            for ev in pats:
                c = ct.calibrate(ev); ops.append(c.total); pk.append(c.peak_factor_size)
            tracemalloc.start(); ct.calibrate(pats[0]); mem = tracemalloc.get_traced_memory()[1]; tracemalloc.stop()
            rows.append(dict(workload=wname, method=f"clique_tree_{h}", n_patterns=len(pats), ops_mean=np.mean(ops),
                             ops_max=max(ops), peak_factor_mean=np.mean(pk), peak_factor_max=max(pk),
                             mem_peak_kb=mem / 1024, preprocessing_ops=pre_ops,
                             max_clique=max(len(x) for x in ct.cliques), n_cliques=len(ct.cliques)))
    return pd.DataFrame(rows)
