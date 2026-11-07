"""Pre-contact data preparation with hard leakage guards (Problem 9)."""
from __future__ import annotations
import numpy as np, pandas as pd

MASKABLE = ["job", "education", "default", "housing", "loan"]


class LeakageError(ValueError):
    pass


def assert_no_leakage(columns, cfg) -> None:
    """Fail loudly if any post-contact (or target) column reaches a model."""
    bad = [c for c in columns
           if any(s in str(c).lower() for s in cfg["forbidden_substrings"])
           or str(c) == cfg["target"]]
    if bad:
        raise LeakageError(f"post-contact/target columns in feature set: {bad}")


def _bins(df, cfg):
    out = pd.DataFrame(index=df.index)
    cuts, labels = cfg["age_bands"]["cuts"], cfg["age_bands"]["labels"]
    out["age_band"] = pd.cut(df["age"], [-np.inf, *cuts, np.inf], right=False, labels=labels).astype(str)
    out["pdays_bin"] = np.where(df["pdays"] == 999, "never",
                         np.where(df["pdays"] <= 6, "0-6", "7+"))
    out["previous_bin"] = pd.cut(df["previous"], [-np.inf, 1, 2, np.inf], right=False,
                                 labels=["0", "1", "2+"]).astype(str)
    for c in ["job", "education", "default", "housing", "loan", "poutcome"]:
        out[c] = df[c].astype(str)
    return out


def prepare(df: pd.DataFrame, cfg: dict):
    """Return dict(split -> frame). Binning uses fixed preregistered cuts, so no
    statistic is fitted on val/test. Columns: BN vars, y, marital group, masks."""
    feats = list(cfg["bn_variables"])
    if cfg["include_macro_columns"]:
        feats += ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    if cfg["include_calendar_columns"]:
        feats += ["month", "day_of_week"]
    if cfg["marital_in_model"]:
        feats += ["marital"]
    assert_no_leakage(feats, cfg)

    X = _bins(df, cfg)
    for c in ("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
              "month", "day_of_week", "marital"):
        if c in feats: X[c] = df[c]
    X = X[feats].copy()
    if cfg["unknown_as_masked"]:
        for c in MASKABLE:
            if c in X: X[f"mask_{c}"] = (X[c] != "unknown").astype(int)   # 1 = observed
    out = X.copy()
    out["y"] = (df[cfg["target"]] == cfg["target_positive"]).astype(int)
    out["marital_group"] = df["marital"].astype(str)
    out["group"] = out["age_band"] + "|" + out["marital_group"]
    assert_no_leakage([c for c in X.columns if not c.startswith("mask_")], cfg)

    n = len(out); s = cfg["split"]
    i1, i2 = int(n * s["train"]), int(n * (s["train"] + s["val"]))
    return {"train": out.iloc[:i1], "val": out.iloc[i1:i2], "test": out.iloc[i2:]}


def group_denominators(frame: pd.DataFrame, min_support: int = 30) -> pd.DataFrame:
    """Per-intersection n, positives (TPR denominator), negatives (FPR denominator)."""
    g = frame.groupby("group")["y"].agg(n="size", pos="sum").reset_index()
    g["neg"] = g["n"] - g["pos"]
    g["exploratory_tpr"] = g["pos"] < min_support
    g["exploratory_fpr"] = g["neg"] < min_support
    return g
