import numpy as np, pandas as pd, pytest, yaml
from src.fixture import fixture
from src.data import prepare, assert_no_leakage, group_denominators, LeakageError

CFG = yaml.safe_load(open("configs/data.yaml"))


def test_duration_never_in_processed_frames():
    for fr in prepare(fixture(), CFG).values():
        assert not any("duration" in c.lower() for c in fr.columns)


def test_guard_rejects_duration_and_target():
    for bad in (["job", "duration"], ["Call_Duration"], ["y"]):
        with pytest.raises(LeakageError):
            assert_no_leakage(bad, CFG)


def test_guard_trips_if_config_adds_duration():
    cfg = dict(CFG, bn_variables=CFG["bn_variables"] + ["duration"])
    with pytest.raises(LeakageError):
        prepare(fixture(), cfg)


def test_declared_exclusions_respected():
    cols = set(prepare(fixture(), CFG)["train"].columns)
    assert not ({"euribor3m", "month", "day_of_week", "marital"} & cols)
    assert "marital_group" in cols      # audit-only column


def test_split_is_positional_and_disjoint():
    s = prepare(fixture(), CFG)
    idx = [set(v.index) for v in s.values()]
    assert not (idx[0] & idx[1]) and not (idx[1] & idx[2])
    assert max(s["train"].index) < min(s["val"].index) < min(s["test"].index)
    assert sum(len(v) for v in s.values()) == 1000


def test_unknown_becomes_masked():
    tr = prepare(fixture(), CFG)["train"]
    assert ((tr["housing"] == "unknown") == (tr["mask_housing"] == 0)).all()


def test_denominators_and_exploratory_flag():
    fr = pd.DataFrame({"group": ["a"] * 40 + ["b"] * 10, "y": [1] * 35 + [0] * 5 + [1] * 2 + [0] * 8})
    d = group_denominators(fr).set_index("group")
    assert d.loc["a", ["n", "pos", "neg"]].tolist() == [40, 35, 5]
    assert d.loc["b", "exploratory_tpr"] and d.loc["a", "exploratory_fpr"]
    assert not d.loc["a", "exploratory_tpr"]
