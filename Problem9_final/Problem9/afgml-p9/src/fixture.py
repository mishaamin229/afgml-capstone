"""SYNTHETIC bank-like table for tests / smoke runs ONLY (never report as real results).
Mimics the column names and coding of bank-additional-full.csv, including a leakage
trap (`duration` correlated with y) and optional group-dependent 'unknown' rates."""
import numpy as np, pandas as pd


def fixture(n=1000, seed=0, young_unknown_boost=0.0):
    r = np.random.default_rng(seed)
    age = np.clip(r.normal(40, 12, n), 18, 90).astype(int)
    pick = lambda xs, p=None: r.choice(xs, n, p=p)
    unk_p = lambda base: np.clip(base + young_unknown_boost * (age < 30), 0, 0.95)
    def with_unknown(vals, base):
        return np.where(r.random(n) < unk_p(base), "unknown", vals)
    poutcome = pick(["nonexistent", "failure", "success"], [0.86, 0.10, 0.04])
    contacted = poutcome != "nonexistent"
    pdays = np.where(contacted, r.integers(0, 21, n), 999)
    previous = np.where(contacted, r.integers(1, 4, n), 0)
    logit = (-2.6 + 2.2 * (poutcome == "success") + 0.5 * contacted
             + 0.4 * ((age >= 60) | (age < 25)) - 0.2 * (pick(["yes", "no"]) == "yes"))
    y = r.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({
        "age": age,
        "job": with_unknown(pick(["admin.", "technician", "blue-collar", "services", "retired", "student"]), 0.01),
        "marital": pick(["married", "single", "divorced", "unknown"], [0.60, 0.28, 0.115, 0.005]),
        "education": with_unknown(pick(["basic.9y", "high.school", "university.degree"]), 0.04),
        "default": with_unknown(pick(["no", "yes"], [0.97, 0.03]), 0.20),
        "housing": with_unknown(pick(["yes", "no"]), 0.02),
        "loan": with_unknown(pick(["yes", "no"], [0.15, 0.85]), 0.02),
        "contact": pick(["cellular", "telephone"]), "month": pick(["may", "jun", "jul"]),
        "day_of_week": pick(["mon", "tue", "wed"]),
        "duration": (np.where(y, 500, 150) * r.exponential(1, n)).astype(int),   # LEAKAGE TRAP
        "campaign": r.integers(1, 10, n), "pdays": pdays, "previous": previous, "poutcome": poutcome,
        "emp.var.rate": r.normal(size=n), "cons.price.idx": r.normal(size=n),
        "cons.conf.idx": r.normal(size=n), "euribor3m": r.normal(size=n),
        "nr.employed": r.normal(size=n), "y": np.where(y, "yes", "no")})
