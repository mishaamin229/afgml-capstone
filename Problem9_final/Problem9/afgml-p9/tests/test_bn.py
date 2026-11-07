import numpy as np, pandas as pd, pytest, yaml
from src.fixture import fixture
from src.data import prepare, LeakageError
from src.bn import fit_bn, predict, verify_cgm_fitted
from src.inference import prior_fallback

DCFG = yaml.safe_load(open("configs/data.yaml"))
MCFG = yaml.safe_load(open("configs/model.yaml"))


@pytest.fixture(scope="module")
def fitted():
    sp = prepare(fixture(1500), DCFG)
    f, s = fit_bn(sp["train"], MCFG["structure"], MCFG["alpha"], DCFG)
    return f, s, sp


def test_cpts_normalized_and_no_duration(fitted):
    f, s, _ = fitted
    assert all(np.allclose(x.values.sum(axis=-1), 1) for x in f)
    assert not any("duration" in v for x in f for v in x.vars)


def test_fit_rejects_leaky_structure(fitted):
    _, _, sp = fitted
    bad = dict(MCFG["structure"], duration=["age_band"])
    with pytest.raises(LeakageError):
        fit_bn(sp["train"], bad, 1.0, DCFG)


def test_cgm_on_fitted_model(fitted):
    f, s, sp = fitted
    r = verify_cgm_fitted(f, s, sp["val"], n_patterns=20)
    assert r["passed"], r


def test_predict_valid_and_budget_triggers_fallback(fitted):
    f, s, sp = fitted
    te = sp["test"].head(60)
    ex = predict(te, f, s)
    assert ex.p1.between(0, 1).all() and not ex.used_fallback.any()
    tight = predict(te, f, s, budget=1, fallback=prior_fallback)
    assert tight.used_fallback.all() and tight.p1.nunique() == 1
    roomy = predict(te, f, s, budget=int(ex.ops.max()))
    assert np.allclose(roomy.p1, ex.p1) and not roomy.used_fallback.any()


def test_synthetic_mask_raises_cost_and_ops_are_deterministic(fitted):
    f, s, sp = fitted
    te = sp["test"].head(60)
    m = pd.DataFrame({v: True for v in ["job", "education", "housing", "loan"]}, index=range(60))
    base, masked = predict(te, f, s), predict(te, f, s, extra_mask=m)
    assert masked.ops.mean() >= base.ops.mean()
    assert (predict(te, f, s).ops.values == base.ops.values).all()
