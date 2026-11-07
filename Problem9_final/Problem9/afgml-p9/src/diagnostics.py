"""Interpretability / graph diagnostics for Problem 9 (all DETERMINISTIC, operation-count based).

* elimination_trace      - step-by-step VE trace: eliminated variable, bucket, induced clique, factor size, ops
* induced_graph_summary  - induced cliques, maximum clique size and treewidth for a heuristic (no evidence)
* clique_tree_tables     - cliques, separators and the two-pass sum-product message schedule
* evidence_pattern_costs - which OBSERVED-SET patterns create the largest intermediate factors
* error_vs_threshold     - do approximation errors / decision flips concentrate near the decision threshold?
"""
from __future__ import annotations
from itertools import combinations
import numpy as np, pandas as pd
from src.inference import (Factor, OpCounter, restrict, product_all, marginalize, _graph, pick_var,
                           variable_elimination, CliqueTree)
from src.bn import evidence_matrix


def elimination_trace(factors, query=("y",), evidence=None, heuristic="min_fill"):
    """Re-run variable elimination while logging every step. Returns (trace DataFrame, posterior,
    total ops). The posterior is asserted equal to `variable_elimination` (the trace is a view of
    the same computation, not a re-implementation of the answer)."""
    evidence = evidence or {}
    ctr = OpCounter()
    work = [restrict(f, evidence) for f in factors]
    keep, rows, step = set(query), [], 0
    while True:
        adj, card = _graph(work)
        cand = [v for v in adj if v not in keep]
        if not cand: break
        v = pick_var(adj, card, cand, heuristic)
        bucket = [f for f in work if v in f.vars]
        rest = [f for f in work if v not in f.vars]
        m0, a0 = ctr.mults, ctr.adds
        prod = product_all(bucket, ctr)
        clique = tuple(prod.vars)                         # the induced clique created by eliminating v
        tau = marginalize(prod, v, ctr)
        step += 1
        rows.append(dict(step=step, eliminated=v, bucket_size=len(bucket), induced_clique=" ".join(map(str, clique)),
                         clique_width=len(clique), product_entries=int(prod.values.size),
                         new_factor_scope=" ".join(map(str, tau.vars)), new_factor_entries=int(tau.values.size),
                         ops_step=(ctr.mults - m0) + (ctr.adds - a0), ops_cum=ctr.total))
        work = rest + [tau]
    joint = product_all(work, ctr); z = joint.values.sum(); ctr.charge("adds", joint.values.size - 1)
    perm = [joint.vars.index(q) for q in query]
    post = joint.values.transpose(perm) / z
    ref, _ = variable_elimination(factors, list(query), evidence, heuristic)
    assert np.allclose(post, ref, atol=1e-12), "trace posterior differs from variable_elimination"
    return pd.DataFrame(rows), post, ctr.total


def induced_graph_summary(factors, heuristics=("min_fill", "min_weight", "min_degree")):
    """Induced cliques / maximum clique size / treewidth of the moralised BN per heuristic
    (no evidence = worst case; evidence only removes variables)."""
    out = []
    for h in heuristics:
        adj, card = _graph(list(factors))
        g = {v: set(n) for v, n in adj.items()}
        cliques, order = [], []
        while g:
            v = pick_var(g, card, list(g), h)
            nb = g[v]; order.append(v); cliques.append(frozenset(nb | {v}))
            for a, b in combinations(nb, 2): g[a].add(b); g[b].add(a)
            for u in nb: g[u].discard(v)
            del g[v]
        maximal = [c for c in set(cliques) if not any(c < d for d in cliques)]
        out.append(dict(heuristic=h, elimination_order=" -> ".join(map(str, order)),
                        max_clique_size=max(len(c) for c in cliques), treewidth=max(len(c) for c in cliques) - 1,
                        n_maximal_cliques=len(maximal),
                        largest_clique_entries=int(max(np.prod([card[v] for v in c]) for c in cliques)),
                        largest_clique=" ".join(sorted(map(str, max(cliques, key=lambda c: np.prod([card[v] for v in c])))))))
    return pd.DataFrame(out)


def clique_tree_tables(ct: CliqueTree, root: int = 0):
    """Cliques, tree edges with separators, and the two-pass (collect -> distribute) message
    schedule used by sum-product calibration. Returns (cliques_df, edges_df, schedule_df)."""
    cl = pd.DataFrame([dict(clique=i, scope=" ".join(sorted(map(str, c))), width=len(c),
                            entries=int(np.prod([ct.card[v] for v in c]))) for i, c in enumerate(ct.cliques)])
    edges = []
    for i, nb in ct.nbrs.items():
        for j in nb:
            if i < j:
                sep = ct.cliques[i] & ct.cliques[j]
                edges.append(dict(i=i, j=j, separator=" ".join(sorted(map(str, sep))),
                                  sep_entries=int(np.prod([ct.card[v] for v in sep])) if sep else 1))
    ed = pd.DataFrame(edges)
    order, seen, stack = [], set(), [(root, None)]
    parent = {}
    while stack:
        x, p = stack.pop()
        if x in seen: continue
        seen.add(x); parent[x] = p; order.append(x)
        stack += [(y, x) for y in ct.nbrs[x] if y not in seen]
    sched = []
    for x in reversed(order):                              # collect: leaves -> root
        if parent[x] is not None: sched.append(("collect", x, parent[x]))
    for x in order:                                        # distribute: root -> leaves
        if parent[x] is not None: sched.append(("distribute", parent[x], x))
    sch = pd.DataFrame([dict(pass_=p, step=k + 1, sender=a, receiver=b,
                             separator=" ".join(sorted(map(str, ct.cliques[a] & ct.cliques[b]))))
                        for k, (p, a, b) in enumerate(sched)])
    return cl, ed, sch


def evidence_pattern_costs(factors, states, df, mask=None, heuristic="min_fill", top=10):
    """Secondary question: which evidence patterns create the largest intermediate factors?
    A pattern = WHICH variables are observed (not their values). Cost is deterministic."""
    vs, E = evidence_matrix(df, states)
    if mask is not None:
        for j, v in enumerate(vs):
            if v in mask.columns: E[mask[v].to_numpy(bool), j] = -1
    obs = (E >= 0)
    keys, inv, counts = np.unique(obs, axis=0, return_inverse=True, return_counts=True)
    rows = []
    for k, pat in enumerate(keys):
        i = int(np.flatnonzero(inv.ravel() == k)[0])
        ev = {v: int(E[i, j]) for j, v in enumerate(vs) if pat[j]}
        c = OpCounter(); variable_elimination(factors, ["y"], ev, heuristic, c)
        rows.append(dict(observed=" ".join(v for j, v in enumerate(vs) if pat[j]) or "(none)",
                         hidden=" ".join(v for j, v in enumerate(vs) if not pat[j]) or "(none)",
                         n_observed=int(pat.sum()), n_queries=int(counts[k]), ops=c.total, peak_factor_entries=c.peak_factor_size))
    d = pd.DataFrame(rows).sort_values(["peak_factor_entries", "ops"], ascending=False)
    return d.head(top).reset_index(drop=True), d


def error_vs_threshold(p_exact, p_approx, thr, n_bins=5):
    """Quantile bins of distance |p_exact - thr| to the decision threshold: mean absolute (= total-variation)
    error and decision-flip rate per bin. If errors concentrate near the threshold, flips follow."""
    pe, pa = np.asarray(p_exact, float), np.asarray(p_approx, float)
    dist = np.abs(pe - thr)
    rank = pd.Series(dist).rank(method="first")
    bins = pd.qcut(rank, n_bins, labels=[f"Q{i + 1}" + (" (closest)" if i == 0 else " (farthest)" if i == n_bins - 1 else "")
                                         for i in range(n_bins)])
    flip = ((pe >= thr) != (pa >= thr)).astype(float)
    return (pd.DataFrame(dict(bin=bins, dist=dist, tv_err=np.abs(pe - pa), flip=flip))
            .groupby("bin", observed=True).agg(n=("flip", "size"), mean_dist=("dist", "mean"),
                                               mean_tv_error=("tv_err", "mean"), flip_rate=("flip", "mean")).reset_index())
