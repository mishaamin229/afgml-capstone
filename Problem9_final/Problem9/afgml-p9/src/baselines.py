"""Predictive baselines (logistic regression, gradient-boosted trees) on the SAME
pre-contact BN variables; fitted on train only."""
from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


def fit_predictive(train, eval_frames: dict, bn_vars, seed=0):
    """Return {name: {split: P(y=1)}} for 'logreg' and 'gbt'."""
    feats = [v for v in bn_vars if v != "y"]
    enc = lambda: ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feats)])
    models = {"logreg": make_pipeline(enc(), LogisticRegression(max_iter=2000)),
              "gbt": make_pipeline(enc(), HistGradientBoostingClassifier(random_state=seed, max_iter=150))}
    out = {}
    for n, m in models.items():
        m.fit(train[feats].astype(str), train["y"])
        out[n] = {k: m.predict_proba(f[feats].astype(str))[:, 1] for k, f in eval_frames.items()}
    return out
