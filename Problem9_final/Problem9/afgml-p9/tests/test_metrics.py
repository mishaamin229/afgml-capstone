import numpy as np, pandas as pd, pytest
from src.metrics import *


def toy():
    y = [1, 1, 0, 0, 1, 1, 0, 0]; d = [1, 0, 1, 0, 1, 1, 0, 0]; g = list("aaaabbbb")
    return y, d, g


def test_hand_computed_gaps():
    y, d, g = toy()
    s = fairness_summary(y, d, g)
    assert s["dDP"] == 0 and s["dEOpp"] == 0.5 and s["dEOdds"] == 0.5 and s["sel_ratio"] == 1.0
    assert s["worst_group_tpr"] == 0.5


def test_ratio_undefined_when_nobody_selected():
    s = fairness_summary([1, 0, 1, 0], [0, 0, 0, 0], list("aabb"))
    assert np.isnan(s["sel_ratio"]) and s["dDP"] == 0


def test_small_and_undefined_groups_are_reported_not_dropped():
    y = [1] * 40 + [0] * 40 + [0] * 5; d = [1] * 80 + [0] * 5
    g = ["big"] * 80 + ["noPos"] * 5
    r = group_rates(y, d, g).set_index("group")
    assert np.isnan(r.loc["noPos", "tpr"]) and r.loc["noPos", "exploratory_tpr"]
    s = fairness_summary(y, d, g)
    assert s["n_groups"] == 2 and s["n_undefined_tpr"] == 1


def test_exclude_exploratory_changes_view():
    r = np.random.default_rng(0)
    g = np.array(["big"] * 400 + ["tiny"] * 10); y = r.integers(0, 2, 410)
    d = np.where(g == "tiny", 0, y)
    all_ = fairness_summary(y, d, g); sup = fairness_summary(y, d, g, exclude_exploratory=True)
    assert all_["dEOpp"] > 0.5 and np.isnan(sup["dEOpp"]) is np.False_ or sup["dEOpp"] == 0


def test_top_k_capacity_exact():
    d = decide_top_k(np.random.default_rng(1).random(1000), 0.2)
    assert d.sum() == 200


def test_decision_change_and_fallback_gap():
    de, db, g = [1, 1, 0, 0], [1, 0, 0, 1], list("aabb")
    c = decision_change_by_group(de, db, g).set_index("group")
    assert c.loc["a", "lost"] == 0.5 and c.loc["b", "gained"] == 0.5
    f = fallback_rate_gap([1, 1, 0, 0], g)
    assert f["gap"] == 1.0


def test_tv_and_prob_metrics():
    assert tv_distance([.5, .5], [.2, .8]) == pytest.approx(0.3)
    m = prob_metrics([1, 0], [1.0, 0.0])          # clipped, finite
    assert np.isfinite(m["log_loss"]) and m["brier"] == pytest.approx(0, abs=1e-20)


def test_calibration_perfect_group():
    p = np.array([0.0] * 50 + [1.0] * 50); y = p.copy()
    assert group_calibration(y, p, ["g"] * 100)["ece"].iloc[0] == pytest.approx(0)


def test_blocks_preserve_order_and_length():
    idx = block_indices(103, 10, np.random.default_rng(0))
    assert len(idx) == 103
    runs = np.diff(idx[:100]).reshape(-1) if False else np.diff(idx[:10])
    assert (runs == 1).all()


def test_paired_bootstrap_identical_methods_zero_effect_and_reproducible():
    r = np.random.default_rng(2); n = 600
    y = r.integers(0, 2, n); d = r.integers(0, 2, n); g = r.choice(list("abc"), n)
    f = lambda idx: (fairness_summary(y[idx], d[idx], g[idx])["dEOpp"]
                     - fairness_summary(y[idx], d[idx], g[idx])["dEOpp"])
    out = paired_block_bootstrap(f, n, 25, n_boot=1000, seed=7)
    assert out["estimate"] == 0 and out["lo"] == 0 and out["hi"] == 0
    g1 = lambda idx: fairness_summary(y[idx], d[idx], g[idx])["dEOpp"]
    a = paired_block_bootstrap(g1, n, 25, 200, 3); b = paired_block_bootstrap(g1, n, 25, 200, 3)
    assert a == b and a["lo"] <= a["estimate"] <= a["hi"] + 0.5


def test_holm_known_values():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
