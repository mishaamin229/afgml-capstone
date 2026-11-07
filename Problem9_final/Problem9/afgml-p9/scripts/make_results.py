"""Figures, LaTeX tables/macros and results/manifest.csv from results/<mode>/metrics."""
import argparse, json, os, subprocess, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); ap.add_argument("--mode", default="real")
ap.add_argument("--config", default="configs/main.yaml")
a = ap.parse_args(); OUT = a.out; M, F, T = f"{OUT}/metrics", f"{OUT}/figures", f"{OUT}/tables"
import yaml; C = yaml.safe_load(open(a.config)); PQ = C["primary_budget_quantile"]
os.makedirs(F, exist_ok=True); os.makedirs(T, exist_ok=True)
agg = pd.read_csv(f"{M}/seed_aggregate.csv"); R = pd.read_csv(f"{M}/main_results.csv")
B = pd.read_csv(f"{M}/bootstrap.csv"); S = pd.read_csv(f"{M}/exp5_aggregate.csv"); X2 = pd.read_csv(f"{M}/exp2_complexity.csv")
G = pd.read_csv(f"{M}/group_table.csv"); log = json.load(open(f"{OUT}/logs/run_log.json"))
tag = "SYNTHETIC SMOKE RUN - NOT EVIDENCE" if a.mode == "smoke" else ""
LAB = {"exact": "Exact (reference)", "global_prior": "Global fallback", "proposed": "Proposed (fair selection)",
       "util_sel": "Utility-only selection", "thr_adj_post_budget": "Threshold adj. after budget", "fair_exact": "Fair-adjusted exact"}

# ----------------------------------------------------- Fig 1: budget curves
for w in ("natural", "dependent"):
    d = agg[agg.workload == w]
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for m in ("exact", "global_prior", "proposed", "util_sel", "thr_adj_post_budget"):
        x = d[d.method == m].sort_values("q")
        for a_, k, yl in zip(ax.ravel(), ("fallback_rate", "worst_group_flip", "ap", "dEOpp_supported"),
                             ("fallback rate", "worst-group decision flip vs exact", "average precision", "ΔEOpp (supported groups)")):
            a_.errorbar(x.budget_ops, x[f"{k}_mean"], yerr=[x[f"{k}_mean"] - x[f"{k}_lo"].fillna(x[f"{k}_mean"]), x[f"{k}_hi"].fillna(x[f"{k}_mean"]) - x[f"{k}_mean"]],
                        marker="o", ms=3, capsize=2, label=LAB[m], alpha=.85); a_.set_ylabel(yl); a_.grid(alpha=.25)
    for a_ in ax[1]: a_.set_xlabel("operation budget per query (mult+add)")
    ax[0, 0].legend(fontsize=7); fig.suptitle(f"Budget–fairness–utility, workload={w} {tag}", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{F}/fig_budget_curves_{w}.png", dpi=160); plt.close(fig)

# ---------------------------------------- Fig 2: fallback rate by group
g = G[(G.method == "global_prior")]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for a_, w in zip(ax, ("natural", "dependent")):
    x = g[g.workload == w].sort_values("group"); col = np.where(x.pos < 30, "#c0504d", "#1f4e79")
    a_.bar(range(len(x)), x.fallback_rate, color=col); a_.set_xticks(range(len(x))); a_.set_xticklabels(x.group, rotation=75, fontsize=6)
    a_.set_title(f"{w} (red: <30 positives, exploratory)", fontsize=9); a_.set_ylabel("fallback rate")
fig.suptitle(f"Fallback rate by age×marital group at primary budget {tag}", fontsize=10); fig.tight_layout()
fig.savefig(f"{F}/fig_group_fallback.png", dpi=160); plt.close(fig)

# --------------------------------- Fig 3: synthetic mechanism (SIMULATED)
s = S[S.fallback == "prior"].sort_values("max_parents")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for a_, k, yl in zip(ax, ("fallback_gap", "dEOpp_degradation"), ("fallback-rate gap between groups", "ΔEOpp degradation vs exact")):
    for nm, lab, c in (("dep", "group-dependent availability", "#c0504d"), ("ind", "group-independent (control)", "#1f4e79")):
        ax_ = a_.errorbar(s.induced_width_mean, s[f"{k}_{nm}_mean"], yerr=[s[f"{k}_{nm}_mean"] - s[f"{k}_{nm}_lo"].fillna(s[f"{k}_{nm}_mean"]), s[f"{k}_{nm}_hi"].fillna(s[f"{k}_{nm}_mean"]) - s[f"{k}_{nm}_mean"]],
                          marker="o", capsize=3, color=c, label=lab)
    a_.set_xlabel("mean induced width (SIMULATED networks)"); a_.set_ylabel(yl); a_.grid(alpha=.25)
ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{F}/fig_synthetic_mechanism.png", dpi=160); plt.close(fig)

# ------------------------------------------------ Fig 4: complexity
p = X2.pivot(index="method", columns="workload", values="ops_mean")
fig, ax = plt.subplots(figsize=(8.5, 3.6)); p.plot.bar(ax=ax); ax.set_ylabel("mean ops per query"); ax.set_xlabel("")
ax.legend(fontsize=6); plt.xticks(rotation=30, ha="right", fontsize=7); fig.tight_layout()
fig.savefig(f"{F}/fig_complexity.png", dpi=160); plt.close(fig)

# --------------------------------------------- tables (csv + LaTeX macros)
nat = agg[(agg.workload == "natural") & (np.isclose(agg.q, PQ))].set_index("method")
body = []
for m in ("exact", "global_prior", "proposed", "util_sel", "thr_adj_post_budget"):
    if m not in nat.index: continue
    r = nat.loc[m]
    body.append(f"{LAB[m]} & {r.ap_mean:.3f} & {r.dEOpp_supported_mean:.3f} & {r.log_loss_mean:.3f} / {r.brier_mean:.3f}\\\\")
open(f"{T}/main_table_body.tex", "w").write("\n".join(body) + "\n")
open(f"{T}/main_table.tex", "w").write("\\begin{tabularx}{\\tablewidth}{Xccc}\n\\toprule\nMethod & AP $\\uparrow$ & $\\Delta_{\\mathrm{EOpp}}$ $\\downarrow$ & Log loss / Brier $\\downarrow$\\\\\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabularx}\n")
nat.to_csv(f"{T}/main_table.csv")
bn = B[(B.workload == "natural") & (B.kind == "proposed_minus_comparator")].set_index("comparison")
fx = lambda c: (f"{bn.loc[c,'estimate']:+.3f} [{bn.loc[c,'lo']:+.3f}, {bn.loc[c,'hi']:+.3f}] (Holm p={bn.loc[c,'p_holm']:.3f})" if c in bn.index else "n/a")
con = pd.read_csv(f"{M}/workload_contrast.csv"); cq = con[(np.isclose(con.q, PQ)) & (con.method == "global_prior")]
syn = S[S.fallback == "prior"]
esc = lambda s: str(s).replace("_", "\\_").replace("%", "\\%")
macros = {
 "resMode": "REAL" if a.mode == "real" else "SMOKE (synthetic fixture; not evidence)",
 "resNtest": log["n"]["test"], "resBudget": int(nat.budget_ops.iloc[0]) if len(nat) else "n/a",
 "resCapacity": C["capacity"], "resSeeds": len(C["seeds"]), "resNboot": int(B.n_boot.iloc[0]) if len(B) else "n/a",
 "resEffectGlobal": fx("proposed - global_prior"), "resEffectUtil": fx("proposed - util_sel"),
 "resEffectThr": fx("proposed - thr_adj_post_budget"),
 "resGapDiff": (f"{cq.fallback_gap_diff_mean.iloc[0]:+.3f} [{cq.fallback_gap_diff_lo.iloc[0]:+.3f}, {cq.fallback_gap_diff_hi.iloc[0]:+.3f}]" if len(cq) else "n/a"),
 "resSynGap": (f"{syn.fallback_gap_diff_mean.mean():+.3f}" if len(syn) else "n/a"),
 "resSynCtrl": (f"{syn.fallback_gap_ind_mean.mean():.3f}" if len(syn) else "n/a"),
 "resCgm": "passed" if all(v["passed"] for v in json.load(open(f"{M}/exp1_exact_agreement.json")).values()) else "FAILED",
 "resCgmErr": f"{max(max(v['max_err_across_orders'], v['max_err_ve_vs_clique_tree'], v['max_err_vs_enumeration']) for v in json.load(open(f'{M}/exp1_exact_agreement.json')).values()):.1e}"}
lines = [f"\\renewcommand{{\\{k}}}{{{esc(v)}}}" for k, v in macros.items()]
lines.append("\\renewcommand{\\resBanner}{" + ("" if a.mode == "real" else "\\textcolor{red}{\\textbf{SMOKE RUN ON SYNTHETIC DATA --- NOT EVIDENCE}}") + "}")
open(f"{T}/macros.tex", "w").write("\n".join(lines) + "\n")

# -------------------------------------------------------------- manifest
try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
except Exception: commit = "no-git"
cmd = {"exp1_exact_agreement.json": "run_experiments.py", "exp2_complexity.csv": "run_experiments.py",
       "main_results.csv": "run_experiments.py", "group_table.csv": "run_experiments.py",
       "chosen_fallbacks.csv": "run_experiments.py", "predictive_baselines.csv": "run_experiments.py",
       "exp5_synthetic.csv": "run_experiments.py", "bootstrap.csv": "evaluate.py", "seed_aggregate.csv": "evaluate.py",
       "workload_contrast.csv": "evaluate.py", "exp5_aggregate.csv": "evaluate.py"}
rows = []
for root, _, files in os.walk(OUT):
    if "smoke" in root.split(os.sep)[1:] and a.mode == "real": continue
    for f in sorted(files):
        pth = os.path.join(root, f)
        rows.append(dict(artifact=pth, command=f"python scripts/{cmd.get(f, 'make_results.py')} (mode={a.mode})",
                         config="configs/main.yaml; configs/data.yaml; configs/model.yaml; configs/evaluation.yaml",
                         seed_or_run=f"seeds={C['seeds']}", data_version=log.get("data_sha256"), code_commit=commit, status="complete"))
pd.DataFrame(rows).to_csv(f"{OUT}/manifest.csv", index=False)
print("make_results done:", len(rows), "artifacts")
