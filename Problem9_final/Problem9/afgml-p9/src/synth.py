"""SYNTHETIC networks with controlled induced width and group-dependent evidence
availability (Exp. 5 mechanism validation). Everything here is simulated; known
structure/labels are NOT facts about any real population."""
from __future__ import annotations
import numpy as np, pandas as pd
from src.inference import (Factor, CliqueTree, variable_elimination, OpCounter, BudgetExceeded)


def gen_network(n_vars=11, max_parents=3, card=3, seed=0):
    """Names: G (binary group, root), X1..Xk, Y (binary leaf). Larger `max_parents`
    -> larger induced width. Returns (factors, cards, order)."""
    rng = np.random.default_rng(seed)
    xs = [f"X{i}" for i in range(1, n_vars - 1)]
    cards = {"G": 2, **{x: card for x in xs}, "Y": 2}
    order, facs = ["G"] + xs + ["Y"], []
    for i, v in enumerate(order):
        pool = order[:i]
        k = 0 if not pool else int(rng.integers(1 if v != "G" else 0, min(max_parents, len(pool)) + 1))
        if v == "Y": k = min(max_parents, len(pool))
        pa = sorted(rng.choice(pool, size=k, replace=False).tolist(), key=order.index) if k else []
        shape = [cards[p] for p in pa] + [cards[v]]
        t = rng.dirichlet(np.full(cards[v], 0.7), size=int(np.prod(shape[:-1]))).reshape(shape)
        facs.append(Factor(tuple(pa + [v]), t))
    return facs, cards, order


def induced_width(factors):
    ct = CliqueTree(factors)
    return max(len(c) for c in ct.cliques) - 1


def sample(factors, cards, order, n, rng):
    cols = {}
    for f in factors:
        child, pa = f.vars[-1], f.vars[:-1]
        probs = f.values[tuple(cols[p] for p in pa)] if pa else np.tile(f.values, (n, 1))
        u = rng.random(n)[:, None]
        cols[child] = (probs.cumsum(1) < u).sum(1).clip(0, cards[child] - 1)
    return pd.DataFrame({v: cols[v] for v in order})


def make_evidence_mask(df, order, avail_by_group, rng):
    """True = variable hidden. G always observed, Y never. avail_by_group[g] = P(observed)."""
    xs = [v for v in order if v not in ("G", "Y")]
    p_obs = np.array([avail_by_group[g] for g in df["G"]])[:, None]
    return pd.DataFrame(rng.random((len(df), len(xs))) >= p_obs, columns=xs)


def _scores(factors, order, df, mask, budget, fallback, simplify=True, heuristic="min_fill"):
    from src.inference import budgeted_query
    xs = [v for v in order if v != "Y"]
    cache, out = {}, []
    for i in range(len(df)):
        ev = {v: int(df.iloc[i][v]) for v in xs if v == "G" or not mask.iloc[i][v]}
        key = tuple(sorted(ev.items()))
        if key not in cache:
            r = budgeted_query(factors, ["Y"], ev, budget, heuristic, simplify, fallback)
            cache[key] = (float(r["posterior"][1]), r["used_fallback"], r["ops"])
        out.append(cache[key])
    return pd.DataFrame(out, columns=["p1", "used_fallback", "ops"])


def run_mechanism(max_parents, dependent: bool, seed, avail=(0.9, 0.5), n_val=1500, n_test=3000,
                  budget_q=0.75, capacity=0.2, n_vars=11):
    """One synthetic run. dependent=True: group 1 observes fewer variables (avail[1]);
    False: BOTH groups use the pooled availability (control: no group-dependent workload).
    Budget is fixed from the INDEPENDENT workload's validation op quantile so both
    workloads face the same deterministic budget."""
    from src.metrics import fairness_summary
    facs, cards, order = gen_network(n_vars, max_parents, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    val, te = sample(facs, cards, order, n_val, rng), sample(facs, cards, order, n_test, rng)
    pooled = float(np.mean(avail))
    av_dep = {0: avail[0], 1: avail[1]}; av_ind = {0: pooled, 1: pooled}
    mk = lambda d, av: make_evidence_mask(d, order, av, rng)
    mv_ind = mk(val, av_ind)
    ops_val = _scores(facs, order, val, mv_ind, None, None)["ops"]
    B = int(np.quantile(ops_val, budget_q))
    av = av_dep if dependent else av_ind
    mv, mt = mk(val, av), mk(te, av)
    ex_v, ex_t = _scores(facs, order, val, mv, None, None), _scores(facs, order, te, mt, None, None)
    thr = float(np.quantile(ex_v.p1, 1 - capacity))
    ypa = [v for v in facs[-1].vars[:-1]]
    res = []
    grp = te["G"].to_numpy(); y = te["Y"].to_numpy()
    d_ex = (ex_t.p1.to_numpy() >= thr).astype(int)
    for name in ("prior", "parents"):
        sc = _scores(facs, order, te, mt, B, _adapt(facs, ypa if name == "parents" else []))
        d = (sc.p1.to_numpy() >= thr).astype(int)
        fr = [sc.used_fallback[grp == g].mean() for g in (0, 1)]
        fl = [(d[grp == g] != d_ex[grp == g]).mean() for g in (0, 1)]
        fs, fe = fairness_summary(y, d, grp), fairness_summary(y, d_ex, grp)
        res.append(dict(max_parents=max_parents, dependent=dependent, seed=seed, fallback=name,
                        induced_width=induced_width(facs), budget_ops=B,
                        fallback_rate_g0=fr[0], fallback_rate_g1=fr[1], fallback_gap=abs(fr[1] - fr[0]),
                        flip_gap=abs(fl[1] - fl[0]), flip_g0=fl[0], flip_g1=fl[1],
                        dEOpp_exact=fe["dEOpp"], dEOpp_budget=fs["dEOpp"],
                        dEOpp_degradation=fs["dEOpp"] - fe["dEOpp"],
                        mean_ops_exact_g0=float(ex_t.ops[grp == 0].mean()),
                        mean_ops_exact_g1=float(ex_t.ops[grp == 1].mean())))
    return res


def _adapt(factors, subset):
    """Fallback for synthetic nets whose target is named 'Y' (VE query on 'Y')."""
    cache = {}
    def f(factors_, query, evidence):
        ev = {v: s for v, s in evidence.items() if v in subset}
        key = tuple(sorted(ev.items()))
        if key not in cache:
            cache[key] = variable_elimination(factors, ["Y"], ev, "min_fill")[0]
        return cache[key]
    return f
