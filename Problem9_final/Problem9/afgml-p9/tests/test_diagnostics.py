import numpy as np, pandas as pd, yaml, pytest
from src.fixture import fixture
from src.data import prepare
from src.bn import fit_bn
from src.inference import CliqueTree, variable_elimination
from src.diagnostics import (elimination_trace, induced_graph_summary, clique_tree_tables,
                             evidence_pattern_costs, error_vs_threshold)
from src.metrics import fast_deopp, supported_groups, fairness_summary
from src.pipeline import run_core

D = yaml.safe_load(open("configs/data.yaml")); M = yaml.safe_load(open("configs/model.yaml"))
C = yaml.safe_load(open("configs/main.yaml"))


@pytest.fixture(scope="module")
def setup():
    sp = prepare(fixture(2500), D); f, s = fit_bn(sp["train"], M["structure"], 1.0, D); return sp, f, s


def test_trace_matches_ve_and_counts_ops(setup):
    sp, f, s = setup
    ev = {"poutcome": 0, "housing": 1}
    for h in ("min_fill", "min_weight", "min_degree"):
        tr, post, ops = elimination_trace(f, ["y"], ev, h)
        ref, _ = variable_elimination(f, ["y"], ev, h)
        assert np.allclose(post, ref) and tr.ops_step.sum() <= ops and len(tr) > 0


def test_induced_graph_and_clique_tree(setup):
    sp, f, s = setup
    g = induced_graph_summary(f)
    assert (g.max_clique_size == g.treewidth + 1).all() and set(g.heuristic) == {"min_fill", "min_weight", "min_degree"}
    cl, ed, sch = clique_tree_tables(CliqueTree(f, "min_fill"))
    assert len(sch) == 2 * len(ed) and len(ed) == len(cl) - 1      # a tree: n-1 edges, 2 messages per edge


def test_pattern_costs_and_threshold_bins(setup):
    sp, f, s = setup
    top, allp = evidence_pattern_costs(f, s, sp["test"], top=3)
    assert len(top) <= 3 and allp.n_queries.sum() == len(sp["test"])
    r = np.random.default_rng(0); pe = r.random(500); pa = np.clip(pe + r.normal(0, .05, 500), 0, 1)
    t = error_vs_threshold(pe, pa, 0.5)
    assert t.n.sum() == 500 and t.flip_rate.iloc[0] >= t.flip_rate.iloc[-1]


def test_fast_deopp_supported_matches_reported_metric():
    r = np.random.default_rng(1)
    g = np.array(["big1"] * 300 + ["big2"] * 300 + ["tiny"] * 3); y = r.integers(0, 2, 603)
    y[-3:] = 1; d = np.where(g == "tiny", 0, r.integers(0, 2, 603))
    codes, uniq = pd.factorize(g); keep = supported_groups(y, codes, len(uniq), 30)
    assert keep.tolist() == [True, True, False]
    assert np.isclose(fast_deopp(y, d, codes, len(uniq), keep), fairness_summary(y, d, g, 30, exclude_exploratory=True)["dEOpp"])
    assert np.isclose(fast_deopp(y, d, codes, len(uniq)), fairness_summary(y, d, g, 30)["dEOpp"])
    assert fast_deopp(y, d, codes, len(uniq)) > fast_deopp(y, d, codes, len(uniq), keep)   # tiny group inflates it


def test_group_unaware_routing_never_uses_protected(setup):
    sp, f, s = setup
    rows, grows, chosen, arts = run_core(sp, f, s, C, D["age_bands"]["labels"], workloads=("natural", "dependent"),
                                         seeds=(0,), budgets={0.75: 11}, collect_q=0.75, tag="t")
    d = pd.DataFrame(rows); ch = pd.DataFrame(chosen)
    assert "proposed_unaware" in set(d.method)
    assert not d[d.method == "proposed_unaware"].group_label_at_decision.any()
    assert not ch.proposed_unaware.isin([n for n, c in C["fallbacks"].items() if c.get("uses_protected")]).any()
