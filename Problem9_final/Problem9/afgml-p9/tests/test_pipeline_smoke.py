import numpy as np, pandas as pd, yaml, pytest
from src.fixture import fixture
from src.data import prepare
from src.bn import fit_bn
from src.pipeline import run_core, natural_budgets, select_fallback
from src.synth import run_mechanism, gen_network, induced_width, sample

D = yaml.safe_load(open("configs/data.yaml")); M = yaml.safe_load(open("configs/model.yaml"))
C = yaml.safe_load(open("configs/main.yaml"))


@pytest.fixture(scope="module")
def setup():
    sp = prepare(fixture(2500), D); f, s = fit_bn(sp["train"], M["structure"], 1.0, D); return sp, f, s


def test_budgets_monotone_and_generous_budget_means_no_fallback(setup):
    sp, f, s = setup
    B = natural_budgets(f, s, sp["val"], "min_fill", C["budget_quantiles"])
    v = list(B.values()); assert v == sorted(v)
    rows, *_ = run_core(sp, f, s, C, D["age_bands"]["labels"], workloads=("natural",), budgets={1.0: 10**9}, tag="t")
    r = pd.DataFrame(rows).set_index("method")
    assert r.loc["global_prior", "fallback_rate"] == 0 and r.loc["proposed", "agree_exact"] == 1.0


def test_selection_rule():
    st = {"a": dict(ap_loss=0.00, worst_flip=0.5, logloss=0.3), "b": dict(ap_loss=0.005, worst_flip=0.1, logloss=0.4),
          "c": dict(ap_loss=0.05, worst_flip=0.0, logloss=0.5)}
    assert select_fallback(st, 0.01, "fair") == "b"     # lowest flip among AP-loss<=tol
    assert select_fallback(st, 0.01, "util") == "a"     # lowest log loss only
    assert select_fallback(st, 0.0001, "fair") == "a"   # none feasible except a (ap_loss 0)


def test_run_core_methods_and_no_duration(setup):
    sp, f, s = setup
    assert not any("duration" in c for c in sp["train"].columns)
    rows, grows, chosen, arts = run_core(sp, f, s, C, D["age_bands"]["labels"], workloads=("natural", "dependent"),
                                         seeds=(0,), budgets={0.75: 11}, collect_q=0.75, tag="t")
    d = pd.DataFrame(rows)
    assert {"exact", "global_prior", "proposed", "util_sel", "thr_adj_post_budget", "fair_exact"} <= set(d.method)
    assert (d[d.method == "exact"].fallback_rate == 0).all()
    assert "d_proposed" in arts["natural"] and len(arts["natural"]["y"]) == len(sp["test"])


def test_synthetic_mechanism_dependent_workload_creates_gap():
    dep = [r["fallback_gap"] for s in range(3) for r in run_mechanism(3, True, s, n_val=500, n_test=1000) if r["fallback"] == "prior"]
    ind = [r["fallback_gap"] for s in range(3) for r in run_mechanism(3, False, s, n_val=500, n_test=1000) if r["fallback"] == "prior"]
    assert np.mean(dep) > np.mean(ind) + 0.1 and np.mean(ind) < 0.1


def test_width_increases_with_max_parents():
    w = {p: np.mean([induced_width(gen_network(11, p, seed=s)[0]) for s in range(5)]) for p in (2, 4)}
    assert w[4] > w[2]
