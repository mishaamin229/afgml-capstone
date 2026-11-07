import numpy as np, pandas as pd, pytest, yaml
from src.fairness import fit_eopp_thresholds, apply_thresholds
from src.metrics import fairness_summary
from src.fixture import fixture
from src.data import prepare
from src.bn import fit_bn, predict
from src.fallbacks import TableFallback

DCFG = yaml.safe_load(open("configs/data.yaml")); MCFG = yaml.safe_load(open("configs/model.yaml"))


def test_eopp_adjustment_reduces_gap_and_matches_capacity():
    r = np.random.default_rng(0); n = 6000
    g = r.choice(["a", "b"], n); y = r.integers(0, 2, n)
    score = r.random(n) * 0.6 + 0.3 * y - 0.15 * (g == "b")
    thr0 = np.quantile(score, 0.8); d0 = (score >= thr0).astype(int)
    th, glob = fit_eopp_thresholds(score, y, g, 0.2)
    d1 = apply_thresholds(score, g, th, glob)
    assert fairness_summary(y, d1, g)["dEOpp"] < fairness_summary(y, d0, g)["dEOpp"]
    assert abs(d1.mean() - 0.2) < 0.01


def test_fallback_table_is_lookup_and_normalized():
    sp = prepare(fixture(2000), DCFG); f, s = fit_bn(sp["train"], MCFG["structure"], 1.0, DCFG)
    fb = TableFallback(f, s, ["poutcome", "pdays_bin"], "small")
    te = sp["test"].head(200)
    a = predict(te, f, s, budget=1, fallback=fb)
    assert a.used_fallback.all() and a.p1.between(0, 1).all()
    prior = TableFallback(f, s, [], "prior")
    b = predict(te, f, s, budget=1, fallback=prior)
    assert b.p1.nunique() == 1 and a.p1.nunique() > 1      # conditioning on a subset helps
