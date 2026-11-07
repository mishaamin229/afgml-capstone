"""C-GM: VE vs clique-tree agreement. Writes results/metrics/cgm.json."""
import argparse, json, os, sys, numpy as np
sys.path.insert(0, ".")
from src.inference import random_bn, variable_elimination, CliqueTree, brute_force, OpCounter

ap = argparse.ArgumentParser(); ap.add_argument("--mode", default="real"); ap.add_argument("--out", default=None)
a = ap.parse_args(); OUT = a.out or ("results" if a.mode == "real" else "results/smoke")
os.makedirs(f"{OUT}/metrics", exist_ok=True)
TOL, rows = 1e-6, []
for seed in range(20):
    cpds, cards = random_bn(n=8, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    names = list(cards)
    cts = {h: CliqueTree(cpds, h) for h in ("min_fill", "min_weight")}
    for case in range(10):
        ev_vars = rng.choice(names, size=int(rng.integers(0, 4)), replace=False).tolist()
        ev = {v: int(rng.integers(cards[v])) for v in ev_vars}
        q = rng.choice([v for v in names if v not in ev], size=1).tolist()
        ref, _ = brute_force(cpds, q, ev)
        for h in cts:
            c = OpCounter(); p, _ = variable_elimination(cpds, q, ev, h, c)
            cc = cts[h].calibrate(ev); pc, _ = cts[h].marginal(q)
            rows.append(dict(seed=seed, case=case, heuristic=h, n_evidence=len(ev),
                             err_ve_vs_enum=float(np.abs(p - ref).max()),
                             err_ct_vs_enum=float(np.abs(pc - ref).max()),
                             err_ve_vs_ct=float(np.abs(p - pc).max()),
                             ve_ops=c.total, ve_peak=c.peak_factor_size,
                             ct_ops=cc.total, ct_peak=cc.peak_factor_size))
mx = max(r["err_ve_vs_ct"] for r in rows)
out = dict(tolerance=TOL, n_comparisons=len(rows), max_abs_err_ve_vs_ct=mx,
           max_abs_err_vs_enum=max(max(r["err_ve_vs_enum"], r["err_ct_vs_enum"]) for r in rows),
           passed=bool(mx <= TOL), rows=rows)
json.dump(out, open(f"{OUT}/metrics/cgm.json", "w"), indent=1)
print(f"C-GM {'PASS' if out['passed'] else 'FAIL'}: n={len(rows)}, max|VE-CT|={mx:.2e}, tol={TOL:.0e}")
