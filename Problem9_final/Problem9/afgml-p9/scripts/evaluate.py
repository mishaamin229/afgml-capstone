"""Statistics: paired block bootstrap, Holm correction, seed aggregation (reads results/<mode>/)."""
import argparse, sys, os, yaml
sys.path.insert(0, ".")
import warnings
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings('ignore', category=RuntimeWarning)
from src.metrics import fast_deopp, paired_block_bootstrap, holm, supported_groups

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/evaluation.yaml"); ap.add_argument("--main-config", default="configs/main.yaml")
ap.add_argument("--out", default="results"); ap.add_argument("--n-boot", type=int, default=None)
a = ap.parse_args()
C = yaml.safe_load(open(a.main_config)); E = yaml.safe_load(open(a.config))
OUT, M = a.out, f"{a.out}/metrics"
nb, bl = a.n_boot or E["bootstrap"]["n_boot"], E["bootstrap"]["block_len"]
fam = C["confirmatory_family"]
mean0 = lambda x: float(np.mean(x)) if len(x) else np.nan
t_ci = lambda x: (lambda n, m, s: (m - stats.t.ppf(0.975, n - 1) * s / np.sqrt(n), m + stats.t.ppf(0.975, n - 1) * s / np.sqrt(n)))(len(x), np.mean(x), np.std(x, ddof=1)) if len(x) > 1 else (np.nan, np.nan)

# -------------------------------------------- paired block bootstrap (fixed models)
rows = []
for w in ("natural", "dependent"):
    p = f"{OUT}/runs/artifacts_{w}.npz"
    if not os.path.exists(p): continue
    A = np.load(p, allow_pickle=True); y = A["y"]; codes, uniq = pd.factorize(A["group"]); nG = len(uniq)
    # Same quantity as the reported dEOpp_supported: groups with >= min_group_support positives
    # in the full test sample (fixed once). 'all_groups' is kept only as a labelled sensitivity.
    keep = supported_groups(y, codes, nG, C["min_group_support"])
    def deo(key, idx, kp=keep): return fast_deopp(y[idx], A[key][idx], codes[idx], nG, kp)
    for other in fam:
        r = paired_block_bootstrap(lambda idx, o=other: deo("d_proposed", idx) - deo(f"d_{o}", idx),
                                   len(y), bl, nb, seed=1)
        rows.append(dict(workload=w, kind="proposed_minus_comparator", comparison=f"proposed - {other}",
                         confirmatory=(w == "natural"), metric="dEOpp, supported groups (negative favours proposed)",
                         n_groups_used=int(keep.sum()), **r))
        r = paired_block_bootstrap(lambda idx, o=other: deo("d_proposed", idx, None) - deo(f"d_{o}", idx, None),
                                   len(y), bl, nb, seed=1)
        rows.append(dict(workload=w, kind="sensitivity_all_groups", comparison=f"proposed - {other}",
                         confirmatory=False, metric="dEOpp, ALL groups incl. exploratory (sensitivity)",
                         n_groups_used=int(nG), **r))
    # exploratory: group-unaware routing vs the primary (possibly protected-attribute) routing
    if "d_proposed_unaware" in A.files:
        for a_, b_ in (("proposed_unaware", "proposed"), ("proposed_unaware", "global_prior")):
            r = paired_block_bootstrap(lambda idx, a_=a_, b_=b_: deo(f"d_{a_}", idx) - deo(f"d_{b_}", idx),
                                       len(y), bl, nb, seed=3)
            rows.append(dict(workload=w, kind="exploratory_unaware", comparison=f"{a_} - {b_}", confirmatory=False,
                             metric="dEOpp, supported groups (negative favours first)", n_groups_used=int(keep.sum()), **r))
    for m in ("global_prior", "proposed", "proposed_unaware", "util_sel", "thr_adj_post_budget"):
        if f"d_{m}" not in A.files: continue
        r = paired_block_bootstrap(lambda idx, k=m: deo(f"d_{k}", idx) - deo("d_exact", idx), len(y), bl, nb, seed=2)
        rows.append(dict(workload=w, kind="degradation_vs_exact", comparison=f"{m} - exact", confirmatory=False,
                         metric="dEOpp degradation, supported groups", n_groups_used=int(keep.sum()), **r))
B = pd.DataFrame(rows)
if len(B):
    conf = B.confirmatory & (B.kind == "proposed_minus_comparator")
    pv = B.loc[conf, "p_two_sided"].fillna(1.0).to_numpy()
    B["p_holm"] = np.nan; B.loc[conf, "p_holm"] = holm(pv)
    B["n_test"] = len(y)
B.to_csv(f"{M}/bootstrap.csv", index=False)

# ------------------------------------------------ seed aggregation (SYNTHETIC masks)
R = pd.read_csv(f"{M}/main_results.csv"); Rm = R[R.tag == "main"]
met = ["ap", "dEOpp_supported", "dEOpp", "dDP", "fallback_rate", "fallback_gap", "worst_group_flip", "agree_exact", "tv_mean", "log_loss", "brier"]
agg = []
for (w, q, m), g in Rm.groupby(["workload", "q", "method"]):
    row = dict(workload=w, q=q, method=m, n_seeds=len(g), budget_ops=g.budget_ops.iloc[0])
    for k in met:
        x = g[k].dropna().to_numpy(); lo, hi = t_ci(x)
        row.update({f"{k}_mean": mean0(x), f"{k}_sd": x.std(ddof=1) if len(x) > 1 else 0.0,
                    f"{k}_lo": lo, f"{k}_hi": hi})
    agg.append(row)
pd.DataFrame(agg).to_csv(f"{M}/seed_aggregate.csv", index=False)

con = []
for (q, m), _ in Rm.groupby(["q", "method"]):
    d = Rm[(Rm.q == q) & (Rm.method == m) & (Rm.workload == "dependent")].set_index("seed")
    i = Rm[(Rm.q == q) & (Rm.method == m) & (Rm.workload == "independent")].set_index("seed")
    if d.empty or i.empty: continue
    row = dict(q=q, method=m, n_seeds=len(d))
    for k in ("fallback_gap", "worst_group_flip", "dEOpp_supported", "ap"):
        x = (d[k] - i[k]).dropna().to_numpy(); lo, hi = t_ci(x)
        row.update({f"{k}_diff_mean": mean0(x), f"{k}_diff_lo": lo, f"{k}_diff_hi": hi})
    con.append(row)
pd.DataFrame(con).to_csv(f"{M}/workload_contrast.csv", index=False)

# --------------------------------------------------------- synthetic networks (Exp 5)
S = pd.read_csv(f"{M}/exp5_synthetic.csv"); ag = []
for (mp, fb), g in S.groupby(["max_parents", "fallback"]):
    dep, ind = g[g.dependent].set_index("seed"), g[~g.dependent].set_index("seed")
    row = dict(max_parents=mp, fallback=fb, induced_width_mean=g.induced_width.mean(), n_seeds=len(dep))
    for k in ("fallback_gap", "flip_gap", "dEOpp_degradation"):
        for nm, d in (("dep", dep), ("ind", ind)):
            x = d[k].dropna().to_numpy(); lo, hi = t_ci(x)
            row.update({f"{k}_{nm}_mean": mean0(x), f"{k}_{nm}_lo": lo, f"{k}_{nm}_hi": hi})
        x = (dep[k] - ind[k]).dropna().to_numpy(); lo, hi = t_ci(x)
        row.update({f"{k}_diff_mean": mean0(x), f"{k}_diff_lo": lo, f"{k}_diff_hi": hi})
    ag.append(row)
pd.DataFrame(ag).to_csv(f"{M}/exp5_aggregate.csv", index=False)
print("evaluate: bootstrap rows", len(B), "| n_boot", nb)
