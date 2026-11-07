# Requirement → evidence matrix

Status key: **ready** = implemented, unit-tested, and exercised on the *synthetic smoke run* (`results/smoke/`, not evidence); **real run pending** until you execute the real-data commands. Update the status column from `results/manifest.csv` after your real run.

| ID | Requirement | Config & command | Artifact / key output | Status | Report/slide |
|---|---|---|---|---|---|
| C-GM | VE vs clique tree within 1e-6 | `configs/evaluation.yaml`; `run_experiments.py` (Exp 1) | `metrics/exp1_exact_agreement.json` | ready; real-data fit pending | §3, S5 |
| C-STAT | Paired uncertainty, valid unit | `evaluate.py` | `metrics/bootstrap.csv` (block bootstrap, 1,000 reps) | ready; real run pending | §3–4 |
| C-LEAK | No post-contact variable | `configs/data.yaml`; `tests/test_data.py`, `test_bn.py` | tests pass; `assert_no_leakage` | complete | S4 |
| C-SPLIT | Positional split, train-only fitting | `prepare_data.py` | `data/processed/*` | ready | S4 |
| C-SEEDS | ≥ 5 seeds for stochastic parts | `configs/main.yaml: seeds` | `main_results.csv`, `seed_aggregate.csv` | ready | S7 |
| C-SMALL | Group sizes, denominators, < 30 flagged | `prepare_data.py`, `run_experiments.py` | `group_denominators.csv`, `group_table.csv` | ready | S4, S8 |
| C-SEL | Preregistered primary metrics, validation selection | `docs/protocol.md` | `chosen_fallbacks.csv` | ready | S6 |
| C-MULT | Holm for confirmatory family | `evaluate.py` | `bootstrap.csv: p_holm` | ready | S8 |
| C-SIM | Simulation labelled | all synthetic outputs under `exp5_*`, `*dependent*` | banners/labels | complete | all |
| P9-E1 | Exact posteriors agree across orders & clique tree | Exp 1 | `exp1_exact_agreement.json` | ready | S5 |
| P9-E2 | Preprocessing/query cost, peak factor, memory by workload | Exp 2 | `exp2_complexity.csv`, `fig_complexity.png` | ready | S11 |
| P9-E3 | Controlled budgets: error, fallback rate, decisions by group | `run_core` sweep | `main_results.csv`, `group_table.csv`, budget-curve figs | ready | S8 |
| P9-E4 | Proposed fallback vs equal-budget alternatives at matched utility | `run_core` (methods) | `bootstrap.csv`, `seed_aggregate.csv` | ready | S8 |
| P9-E5 | Synthetic treewidth × availability incl. no-group-dependence control | Exp 5 | `exp5_synthetic.csv`, `exp5_aggregate.csv`, `fig_synthetic_mechanism.png` | ready | S9 |
| P9-A1 | Remove fairness-aware fallback selection | method `util_sel` | `main_results.csv` | ready | S9 |
| P9-A2 | One global fallback | method `global_prior` | `main_results.csv` | ready | S9 |
| P9-A3 | Disable evidence-based simplification | tag `ablation_no_simplification` | `main_results.csv` | ready | S9 |
| P9-A4 | Generous budgets ≈ unrestricted | budget level q = 1.00 | `main_results.csv` | ready | S9 |
| P9-R1 | Different elimination orders | tags `robust_heuristic_*` | `main_results.csv` | ready | S9 |
| P9-R2 | Network densities | tags `robust_structure_{sparse,dense}` | `main_results.csv` | ready | S9 |
| P9-R3 | Discretisations | tag `robust_discretization` | `main_results.csv` | ready | S9 |
| P9-R4 | Operation budgets | q ∈ {.5,.75,.9,.95,1} | `main_results.csv` | ready | S8 |
| P9-R5 | Evidence masks | workloads natural/dependent/independent | `main_results.csv` | ready | S8 |
| P9-R6 | Decision thresholds | tags `robust_capacity_*` | `main_results.csv` | ready | S9 |
| P9-R7 | Wall-clock separately, repeated batches | `profile_latency.py` | `metrics/latency.csv` | ready | S12 |
| P9-B | Predictive baselines (LR, GBT); fairness baselines | `run_experiments.py` | `predictive_baselines.csv` | ready | S7 |
| D-1…7 | Proposal, related work, cards, assumptions, repo, report, demo | `docs/`, `report/`, `slides/`, `demo/` | files present | ready; **students must finalise names, dates, contributions, literature** | — |
| P9-RT | Group-unaware routing comparison (brief: if protected attributes are used in routing) | `run_core` method `proposed_unaware` | `main_results.csv`, `bootstrap.csv` (exploratory) | ready; real run pending | S8 |
| P9-IG | Elimination traces, induced cliques, max clique size, evidence propagation | `src/diagnostics.py`; notebook §4 | notebook output | ready | S5 |
| P9-S2 | Secondary questions: largest factors by evidence pattern; errors near the threshold | `evidence_pattern_costs`, `error_vs_threshold`; notebook §6, §12 | notebook output | ready | S11 |
| P9-E5b | Treewidth × availability-gap factorial (SIMULATED) | notebook §13 | `metrics/exp5b_factorial.csv` | ready | S9 |
