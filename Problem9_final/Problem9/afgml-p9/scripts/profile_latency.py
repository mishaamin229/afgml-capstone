"""Wall-clock latency, measured SEPARATELY from the deterministic op-count budgets
(repeated timed batches; hardware dependent; never used to define a budget)."""
import argparse, json, os, platform, sys, time
sys.path.insert(0, ".")
import numpy as np, pandas as pd, yaml
from src.data import prepare
from src.bn import fit_bn, evidence_matrix
from src.inference import variable_elimination, CliqueTree, OpCounter

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["real", "smoke"], default="real"); ap.add_argument("--out", default=None)
a = ap.parse_args()
C, D, M = (yaml.safe_load(open(p)) for p in ("configs/main.yaml", "configs/data.yaml", "configs/model.yaml"))
OUT = a.out or ("results" if a.mode == "real" else "results/smoke"); os.makedirs(f"{OUT}/metrics", exist_ok=True)
if a.mode == "real":
    df = pd.read_csv(D["source_file"], sep=D["sep"])
else:
    from src.fixture import fixture; df = fixture(3000)
sp = prepare(df, D); f, s = fit_bn(sp["train"], M["structure"], M["alpha"], D)
vs, E = evidence_matrix(sp["test"], s); rng = np.random.default_rng(0)
nq, nb = C["latency"]["queries_per_batch"], C["latency"]["batches"]
ct = {h: CliqueTree(f, h) for h in ("min_fill", "min_weight")}
rows = []
for name, fn in [(f"ve_{h}", (lambda ev, h=h: variable_elimination(f, ["y"], ev, h))) for h in ("min_fill", "min_weight", "min_degree")] + \
                [(f"clique_tree_{h}", (lambda ev, h=h: (ct[h].calibrate(ev), ct[h].marginal(["y"])))) for h in ct]:
    per_batch = []
    for b in range(nb):
        idx = rng.choice(len(E), nq)
        evs = [{v: int(c) for v, c in zip(vs, E[i]) if c >= 0} for i in idx]
        t = time.perf_counter()
        for ev in evs: fn(ev)
        per_batch.append((time.perf_counter() - t) / nq * 1e3)
    rows.append(dict(method=name, ms_per_query_median=float(np.median(per_batch)),
                     ms_q1=float(np.quantile(per_batch, .25)), ms_q3=float(np.quantile(per_batch, .75)),
                     batches=nb, queries_per_batch=nq, python=platform.python_version(), machine=platform.platform()))
pd.DataFrame(rows).to_csv(f"{OUT}/metrics/latency.csv", index=False)
print(pd.DataFrame(rows)[["method", "ms_per_query_median", "ms_q1", "ms_q3"]].round(3).to_string(index=False))
