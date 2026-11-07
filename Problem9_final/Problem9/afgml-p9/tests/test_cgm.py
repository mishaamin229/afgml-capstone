import numpy as np, pytest
from src.inference import (random_bn, variable_elimination, CliqueTree,
                           brute_force, OpCounter, product_all)

TOL = 1e-6
HEUR = ["min_fill", "min_weight"]


def _cases(cards, rng, k=6):
    names = list(cards)
    for _ in range(k):
        m = int(rng.integers(0, 4))
        ev_vars = rng.choice(names, size=m, replace=False).tolist()
        ev = {v: int(rng.integers(cards[v])) for v in ev_vars}
        q = rng.choice([v for v in names if v not in ev], size=1).tolist()
        yield q, ev


@pytest.mark.parametrize("seed", range(15))
def test_cpds_normalized(seed):
    cpds, _ = random_bn(seed=seed)
    for f in cpds:
        assert np.allclose(f.values.sum(axis=-1), 1.0, atol=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_joint_sums_to_one(seed):
    cpds, _ = random_bn(seed=seed)
    assert abs(product_all(cpds).values.sum() - 1.0) < 1e-9


@pytest.mark.parametrize("seed", range(15))
def test_ve_vs_clique_tree_vs_enumeration(seed):
    cpds, cards = random_bn(seed=seed)
    rng = np.random.default_rng(1000 + seed)
    ct = {h: CliqueTree(cpds, h) for h in HEUR}
    for q, ev in _cases(cards, rng):
        ref, zref = brute_force(cpds, q, ev)
        outs = []
        for h in HEUR:
            p, z = variable_elimination(cpds, q, ev, h)
            outs.append((p, z))
            ct[h].calibrate(ev)
            outs.append(ct[h].marginal(q))
        for p, z in outs:
            assert np.max(np.abs(p - ref)) <= TOL
            assert abs(z - zref) <= TOL * max(1.0, zref)


def test_order_changes_cost_not_posterior():
    cpds, cards = random_bn(n=9, seed=3)
    q, ev = ["V8"], {"V0": 0}
    c1, c2 = OpCounter(), OpCounter()
    p1, _ = variable_elimination(cpds, q, ev, "min_fill", c1)
    p2, _ = variable_elimination(cpds, q, ev, "min_weight", c2)
    assert np.max(np.abs(p1 - p2)) <= TOL   # same posterior
    assert c1.total > 0 and c2.total > 0    # costs are recorded (may differ)


# ------------------------------------------------ budgets & simplification
from src.inference import (BudgetExceeded, budgeted_query, prune_barren,
                           prior_fallback, make_subset_fallback)


@pytest.mark.parametrize("seed", range(10))
def test_barren_pruning_is_exact(seed):
    cpds, cards = random_bn(n=9, seed=seed)
    ref, _ = brute_force(cpds, ["V8"], {"V0": 0})
    pruned = prune_barren(cpds, ["V8"], {"V0": 0})
    p, _ = variable_elimination(pruned, ["V8"], {"V0": 0})
    assert np.max(np.abs(p - ref)) <= TOL and len(pruned) <= len(cpds)


def test_budget_is_deterministic_and_enforced():
    cpds, _ = random_bn(n=9, seed=4)
    free = budgeted_query(cpds, ["V8"], {}, budget=None)
    again = budgeted_query(cpds, ["V8"], {}, budget=None)
    assert free["ops"] == again["ops"] and not free["used_fallback"]
    with pytest.raises(BudgetExceeded):
        budgeted_query(cpds, ["V8"], {}, budget=max(1, free["ops"] // 2))
    ok = budgeted_query(cpds, ["V8"], {}, budget=free["ops"])      # exactly enough
    assert not ok["used_fallback"]
    assert np.max(np.abs(ok["posterior"] - free["posterior"])) <= TOL


def test_fallbacks_return_valid_distributions():
    cpds, _ = random_bn(n=9, seed=4)
    for fb in (prior_fallback, make_subset_fallback(["V0", "V1"], budget=10**6)):
        r = budgeted_query(cpds, ["V8"], {"V0": 0, "V5": 1}, budget=1, fallback=fb)
        assert r["used_fallback"] and abs(r["posterior"].sum() - 1) < 1e-9
