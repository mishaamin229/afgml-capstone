"""Exact inference for discrete Bayesian networks: factor ops, variable
elimination (VE) with pluggable heuristics, and clique-tree calibration.

Every factor operation increments an OpCounter. Budgets are defined on these
deterministic counts (never wall-clock time).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np


class BudgetExceeded(Exception):
    """Raised BEFORE an operation that would push the op count past the budget."""


@dataclass
class OpCounter:
    mults: int = 0          # one per output entry of a factor product
    adds: int = 0           # additions performed while summing out
    peak_factor_size: int = 0
    budget: int | None = None   # deterministic cap on mults+adds (None = unlimited)

    def charge(self, kind: str, n: int) -> None:
        if self.budget is not None and self.total + n > self.budget:
            raise BudgetExceeded(f"{self.total}+{n} > {self.budget}")
        setattr(self, kind, getattr(self, kind) + int(n))

    @property
    def total(self) -> int:
        return self.mults + self.adds

    def see(self, size: int) -> None:
        self.peak_factor_size = max(self.peak_factor_size, int(size))


@dataclass
class Factor:
    vars: tuple
    values: np.ndarray

    def __post_init__(self):
        self.vars = tuple(self.vars)
        self.values = np.asarray(self.values, dtype=float)
        assert self.values.ndim == len(self.vars), "scope/array rank mismatch"
        assert len(set(self.vars)) == len(self.vars), "repeated variable in scope"

    def card(self, v):
        return self.values.shape[self.vars.index(v)]


def _align(f: Factor, order: list) -> np.ndarray:
    idx = sorted(range(len(f.vars)), key=lambda i: order.index(f.vars[i]))
    arr = f.values.transpose(idx)
    shape, k = [], 0
    for v in order:
        if v in f.vars:
            shape.append(arr.shape[k]); k += 1
        else:
            shape.append(1)
    return arr.reshape(shape)


def _is_identity(f: Factor) -> bool:
    return f.vars == () and float(f.values) == 1.0


def multiply(f: Factor, g: Factor, ctr: OpCounter | None = None) -> Factor:
    if _is_identity(f): return g          # multiplicative identity: free
    if _is_identity(g): return f
    order = list(f.vars) + [v for v in g.vars if v not in f.vars]
    if ctr is not None:
        cards = {**dict(zip(f.vars, f.values.shape)), **dict(zip(g.vars, g.values.shape))}
        size = int(np.prod([cards[v] for v in order]))
        ctr.charge("mults", size)
        ctr.see(size)
    return Factor(tuple(order), _align(f, order) * _align(g, order))


def marginalize(f: Factor, v, ctr: OpCounter | None = None) -> Factor:
    ax = f.vars.index(v)
    if ctr is not None:
        ctr.charge("adds", f.values.size - f.values.size // f.values.shape[ax])
    out = f.values.sum(axis=ax)
    return Factor(tuple(x for x in f.vars if x != v), out)


def restrict(f: Factor, evidence: dict) -> Factor:
    """Condition on evidence by slicing (drops evidence variables from scope)."""
    vals, vs = f.values, list(f.vars)
    for v, s in evidence.items():
        if v in vs:
            ax = vs.index(v)
            vals = np.take(vals, s, axis=ax)
            vs.pop(ax)
    return Factor(tuple(vs), vals)


def product_all(factors, ctr=None) -> Factor:
    out = Factor((), np.array(1.0))
    for f in factors:
        out = multiply(out, f, ctr)
    return out


# ---------------------------------------------------------------- heuristics
def _graph(factors):
    adj, card = {}, {}
    for f in factors:
        for v in f.vars:
            adj.setdefault(v, set()); card[v] = f.card(v)
        for a, b in combinations(f.vars, 2):
            adj[a].add(b); adj[b].add(a)
    return adj, card


def _fill(adj, v):
    nb = sorted(adj[v], key=str)
    return sum(1 for a, b in combinations(nb, 2) if b not in adj[a])


def _weight(adj, card, v):
    w = 1
    for u in adj[v]:
        w *= card[u]
    return w


def pick_var(adj, card, candidates, heuristic: str):
    key = {"min_fill": lambda v: (_fill(adj, v), _weight(adj, card, v), str(v)),
           "min_weight": lambda v: (_weight(adj, card, v), _fill(adj, v), str(v)),
           "min_degree": lambda v: (len(adj[v]), _fill(adj, v), str(v))}[heuristic]
    return min(candidates, key=key)


# ------------------------------------------------------ variable elimination
def variable_elimination(factors, query, evidence=None, heuristic="min_fill",
                         ctr: OpCounter | None = None):
    """Return (P(query | evidence) as array in `query` order, P(evidence))."""
    evidence = evidence or {}
    ctr = ctr if ctr is not None else OpCounter()
    work = [restrict(f, evidence) for f in factors]
    keep = set(query)
    while True:
        adj, card = _graph(work)
        cand = [v for v in adj if v not in keep]
        if not cand:
            break
        v = pick_var(adj, card, cand, heuristic)
        bucket = [f for f in work if v in f.vars]
        rest = [f for f in work if v not in f.vars]
        tau = marginalize(product_all(bucket, ctr), v, ctr)
        work = rest + [tau]
    joint = product_all(work, ctr)
    z = joint.values.sum()
    ctr.charge("adds", joint.values.size - 1)
    perm = [joint.vars.index(q) for q in query]
    return joint.values.transpose(perm) / z, float(z)


# --------------------------------------------------------------- clique tree
class CliqueTree:
    def __init__(self, factors, heuristic="min_fill"):
        self.factors = list(factors)
        adj, self.card = _graph(self.factors)
        cliques, g = [], {v: set(n) for v, n in adj.items()}
        while g:
            v = pick_var(g, self.card, list(g), heuristic)
            nb = g[v]
            cliques.append(frozenset(nb | {v}))
            for a, b in combinations(nb, 2):
                g[a].add(b); g[b].add(a)
            for u in nb:
                g[u].discard(v)
            del g[v]
        self.cliques = [c for c in set(cliques)
                        if not any(c < d for d in cliques)]
        self.cliques.sort(key=lambda c: sorted(map(str, c)))
        n = len(self.cliques)
        edges = sorted(((len(self.cliques[i] & self.cliques[j]), i, j)
                        for i, j in combinations(range(n), 2)), key=lambda e: (-e[0], e[1], e[2]))
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        self.nbrs = {i: [] for i in range(n)}
        for _, i, j in edges:
            if find(i) != find(j):
                parent[find(i)] = find(j)
                self.nbrs[i].append(j); self.nbrs[j].append(i)
        # running-intersection check
        self._check_rip()

    def _check_rip(self):
        for v in {x for c in self.cliques for x in c}:
            nodes = {i for i, c in enumerate(self.cliques) if v in c}
            seen, stack = set(), [next(iter(nodes))]
            while stack:
                x = stack.pop()
                if x in seen: continue
                seen.add(x)
                stack += [y for y in self.nbrs[x] if y in nodes]
            assert seen == nodes, f"running intersection violated for {v}"

    def calibrate(self, evidence=None, ctr: OpCounter | None = None):
        evidence = evidence or {}
        ctr = ctr if ctr is not None else OpCounter()
        n = len(self.cliques)
        psi = [Factor((), np.array(1.0)) for _ in range(n)]
        def home(scope):
            return next(i for i, c in enumerate(self.cliques) if set(scope) <= c)
        for f in self.factors:
            i = home(f.vars); psi[i] = multiply(psi[i], f, ctr)
        for v, s in evidence.items():
            ind = np.zeros(self.card[v]); ind[s] = 1.0
            i = home((v,)); psi[i] = multiply(psi[i], Factor((v,), ind), ctr)
        memo = {}
        def msg(i, j):
            if (i, j) in memo: return memo[(i, j)]
            f = psi[i]
            for k in self.nbrs[i]:
                if k != j: f = multiply(f, msg(k, i), ctr)
            for v in [v for v in f.vars if v not in self.cliques[j]]:
                f = marginalize(f, v, ctr)
            memo[(i, j)] = f
            return f
        self.beliefs = []
        for i in range(n):
            b = psi[i]
            for k in self.nbrs[i]: b = multiply(b, msg(k, i), ctr)
            self.beliefs.append(b)
        self.ctr = ctr
        return ctr

    def marginal(self, query):
        i = next(i for i, c in enumerate(self.cliques) if set(query) <= c)
        b = self.beliefs[i]
        for v in [v for v in b.vars if v not in query]:
            b = marginalize(b, v)
        z = b.values.sum()
        perm = [b.vars.index(q) for q in query]
        return b.values.transpose(perm) / z, float(z)


# ------------------------------------------------------------ test utilities
def random_bn(n=8, max_parents=3, card_range=(2, 4), seed=0):
    """Random discrete BN as a list of CPD factors (parents..., child)."""
    rng = np.random.default_rng(seed)
    names = [f"V{i}" for i in range(n)]
    cards = {v: int(rng.integers(card_range[0], card_range[1] + 1)) for v in names}
    cpds = []
    for i, v in enumerate(names):
        k = int(rng.integers(0, min(max_parents, i) + 1))
        pa = sorted(rng.choice(i, size=k, replace=False).tolist()) if k else []
        pa = [names[p] for p in pa]
        shape = [cards[p] for p in pa] + [cards[v]]
        t = rng.dirichlet(np.ones(cards[v]), size=int(np.prod(shape[:-1])) or 1).reshape(shape)
        cpds.append(Factor(tuple(pa + [v]), t))
    return cpds, cards


def brute_force(factors, query, evidence):
    j = product_all([restrict(f, evidence) for f in factors])
    for v in [v for v in j.vars if v not in query]:
        j = marginalize(j, v)
    z = j.values.sum()
    perm = [j.vars.index(q) for q in query]
    return j.values.transpose(perm) / z, float(z)


# ------------------------------------------- evidence-based simplification
def prune_barren(factors, query, evidence):
    """Exact simplification: drop CPDs of nodes that are not ancestors of
    query U evidence (child = last variable in each CPD scope). Posterior unchanged."""
    child = {f.vars[-1]: f for f in factors}
    need, stack = set(), list(query) + list(evidence)
    while stack:
        v = stack.pop()
        if v in need: continue
        need.add(v)
        stack += list(child[v].vars[:-1])
    return [f for f in factors if f.vars[-1] in need]


def budgeted_query(factors, query, evidence, budget, heuristic="min_fill",
                   simplify=True, fallback=None):
    """Exact VE under a deterministic op budget. On BudgetExceeded call
    `fallback(factors, query, evidence)` (-> posterior array) if given.
    Returns dict(posterior, used_fallback, ops, peak)."""
    ctr = OpCounter(budget=budget)
    fs = prune_barren(factors, query, evidence) if simplify else factors
    try:
        p, _ = variable_elimination(fs, query, evidence, heuristic, ctr)
        return dict(posterior=p, used_fallback=False, ops=ctr.total, peak=ctr.peak_factor_size)
    except BudgetExceeded:
        if fallback is None: raise
        return dict(posterior=fallback(factors, query, evidence), used_fallback=True,
                    ops=ctr.total, peak=ctr.peak_factor_size)


def prior_fallback(factors, query, evidence):
    """Cheapest fallback: ignore all evidence (precompute offline in practice)."""
    return variable_elimination(factors, query, {}, "min_fill")[0]


def make_subset_fallback(keep_vars, budget, heuristic="min_fill"):
    """Fallback that conditions only on evidence in `keep_vars`; if even that
    exceeds `budget`, returns the prior."""
    keep = set(keep_vars)
    def fb(factors, query, evidence):
        ev = {v: s for v, s in evidence.items() if v in keep}
        try:
            return budgeted_query(factors, query, ev, budget, heuristic)["posterior"]
        except BudgetExceeded:
            return prior_fallback(factors, query, evidence)
    return fb
