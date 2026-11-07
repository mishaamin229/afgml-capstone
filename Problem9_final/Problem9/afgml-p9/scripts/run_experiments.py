"""Run every Problem-9 experiment. `--mode real` uses data/raw/bank-additional-full.csv;
`--mode smoke` uses a SYNTHETIC bank-like fixture (pipeline check only; not evidence)."""
import argparse, copy, hashlib, json, platform, sys, time, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from src.data import prepare
from src.bn import fit_bn
from src.pipeline import run_core, natural_budgets, make_mask, eval_method
from src.profiling import exact_agreement, complexity_profile
from src.baselines import fit_predictive
from src.synth import run_mechanism
from src.bn import predict

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/main.yaml"); ap.add_argument("--data-config", default="configs/data.yaml")
ap.add_argument("--model-config", default="configs/model.yaml")
ap.add_argument("--mode", choices=["real", "smoke"], default="real"); ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default=None); ap.add_argument("--n-smoke", type=int, default=8000)
a = ap.parse_args()
C, D, M = (yaml.safe_load(open(p)) for p in (a.config, a.data_config, a.model_config))
OUT = a.out or ("results" if a.mode == "real" else "results/smoke")
for sub in ("metrics", "runs", "logs", "figures", "tables"): os.makedirs(f"{OUT}/{sub}", exist_ok=True)
if a.quick: C["seeds"] = C["seeds"][:2]; C["synthetic"]["widths"] = C["synthetic"]["widths"][:2]
t0, log = time.time(), {"mode": a.mode, "quick": a.quick}

# ------------------------------------------------------------------ data
if a.mode == "real":
    if not os.path.exists(D["source_file"]):
        sys.exit(f"Missing {D['source_file']}. Run: python scripts/acquire_data.py")
    df = pd.read_csv(D["source_file"], sep=D["sep"])
    log["data_sha256"] = hashlib.sha256(open(D["source_file"], "rb").read()).hexdigest()
else:
    from src.fixture import fixture
    df = fixture(2500 if a.quick else a.n_smoke, seed=0, young_unknown_boost=0.0)
    log["data_sha256"] = "SYNTHETIC-FIXTURE"
splits = prepare(df, D); bl = D["age_bands"]["labels"]
factors, states = fit_bn(splits["train"], M["structure"], M["alpha"], D)
log["n"] = {k: len(v) for k, v in splits.items()}
from src.data import group_denominators
pd.concat([group_denominators(v, C["min_group_support"]).assign(split=k) for k, v in splits.items()]).to_csv(
    f"{OUT}/metrics/group_denominators.csv", index=False)
print("data", log["n"], flush=True)

# --------------------------------------------- Exp 1: exact agreement (C-GM)
mask_nat = None
mdep = make_mask(splits["test"], C, bl, "dependent", 0, 1)
agree = {"natural": exact_agreement(factors, states, splits["test"], None, 60 if a.quick else 300, tol=1e-6),
         "synthetic_dependent_mask": exact_agreement(factors, states, splits["test"], mdep, 60 if a.quick else 300, tol=1e-6)}
json.dump(agree, open(f"{OUT}/metrics/exp1_exact_agreement.json", "w"), indent=1)
print("C-GM", {k: v["passed"] for k, v in agree.items()}, flush=True)

# ------------------------------------------------- Exp 2: complexity profile
complexity_profile(factors, states, splits["test"], C["mask_vars"], cap=60 if a.quick else 200).to_csv(
    f"{OUT}/metrics/exp2_complexity.csv", index=False)

# ------------------------------------------------ Exp 3-4 + ablations: main
H = C["heuristic"]; Q = C["budget_quantiles"]; PQ = C["primary_budget_quantile"]
B0 = natural_budgets(factors, states, splits["val"], H, Q)
log["budgets_ops"] = B0
R, G, CH, ART = [], [], [], {}
def go(**kw):
    r, g, c, ar = run_core(**kw); R.extend(r); G.extend(g); CH.extend(c); ART.update(ar); print("done", kw.get("tag"), round(time.time() - t0), "s", flush=True)
base = dict(splits=splits, factors=factors, states=states, cfg=C, band_labels=bl, seeds=tuple(C["seeds"]))
go(**base, budgets=B0, collect_q=PQ, tag="main", heuristic=H)
go(**base, budgets=B0, tag="ablation_no_simplification", simplify=False, workloads=("natural", "dependent"))
for h in C["heuristics_compare"]:
    go(**{**base, "seeds": tuple(C["seeds"])}, budgets=natural_budgets(factors, states, splits["val"], h, [PQ]),
       tag=f"robust_heuristic_{h}", heuristic=h, workloads=("natural", "dependent"))
for cap in C["capacities_robust"]:
    go(**{**base, "seeds": tuple(C["seeds"])}, budgets={PQ: B0[PQ]}, tag=f"robust_capacity_{cap}", capacity=cap,
       workloads=("natural", "dependent"))
# alternative discretisation: refit with different age bands, budgets recomputed
D2 = copy.deepcopy(D); D2["age_bands"] = C["robust_data"]["age_bands"]
sp2 = prepare(df, D2); f2, s2 = fit_bn(sp2["train"], M["structure"], M["alpha"], D2)
go(splits=sp2, factors=f2, states=s2, cfg=C, band_labels=D2["age_bands"]["labels"], seeds=tuple(C["seeds"]),
   budgets=natural_budgets(f2, s2, sp2["val"], H, [PQ]), tag="robust_discretization", workloads=("natural", "dependent"))
# alternative network densities
for name, ov in C["robust_structures"].items():
    st = {**M["structure"], **ov}
    f3, s3 = fit_bn(splits["train"], st, M["alpha"], D)
    go(splits=splits, factors=f3, states=s3, cfg=C, band_labels=bl, seeds=tuple(C["seeds"]),
       budgets=natural_budgets(f3, s3, splits["val"], H, [PQ]), tag=f"robust_structure_{name}", workloads=("natural", "dependent"))
pd.DataFrame(R).to_csv(f"{OUT}/metrics/main_results.csv", index=False)
pd.DataFrame(CH).to_csv(f"{OUT}/metrics/chosen_fallbacks.csv", index=False)
pd.concat(G).to_csv(f"{OUT}/metrics/group_table.csv", index=False)
for w, d in ART.items(): np.savez_compressed(f"{OUT}/runs/artifacts_{w}.npz", **d)

# ------------------------------------------- predictive baselines (LR, GBT)
ex_v = predict(splits["val"], factors, states); ex_t = predict(splits["test"], factors, states)
thr_ex = float(np.quantile(ex_v.p1, 1 - C["capacity"])); d_ex = (ex_t.p1.to_numpy() >= thr_ex).astype(int)
rows = []
for seed in C["seeds"]:
    pr = fit_predictive(splits["train"], {"val": splits["val"], "test": splits["test"]}, D["bn_variables"], seed)
    for name, p in pr.items():
        thr = float(np.quantile(p["val"], 1 - C["capacity"])); d = (p["test"] >= thr).astype(int)
        rows.append(dict(method=name, seed=seed, **eval_method(splits["test"].y.to_numpy(), p["test"], d,
                         splits["test"].group.to_numpy(), d_ex, ex_t.p1.to_numpy(), np.zeros(len(d), bool), C["min_group_support"])))
r0 = eval_method(splits["test"].y.to_numpy(), ex_t.p1.to_numpy(), d_ex, splits["test"].group.to_numpy(), d_ex,
                 ex_t.p1.to_numpy(), np.zeros(len(d_ex), bool), C["min_group_support"])
rows.append(dict(method="bn_exact", seed=0, **r0))
pd.DataFrame(rows).to_csv(f"{OUT}/metrics/predictive_baselines.csv", index=False)

# ----------------------------------- Exp 5: synthetic networks (SIMULATED)
S = C["synthetic"]; srows = []
for w in S["widths"]:
    for dep in (True, False):
        for seed in C["seeds"]:
            srows += run_mechanism(w, dep, seed, tuple(S["availability"]), n_val=800 if a.quick else 1500,
                                   n_test=1500 if a.quick else 3000, budget_q=PQ, capacity=C["capacity"], n_vars=S["n_vars"])
pd.DataFrame(srows).to_csv(f"{OUT}/metrics/exp5_synthetic.csv", index=False)

log.update(runtime_s=round(time.time() - t0), python=platform.python_version(), platform=platform.platform(),
           seeds=C["seeds"], synthetic_label="SYNTHETIC masks/networks are simulated, not facts about customers" )
json.dump(log, open(f"{OUT}/logs/run_log.json", "w"), indent=1)
print("finished", log["runtime_s"], "s")
