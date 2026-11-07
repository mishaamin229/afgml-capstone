# Fairness Preservation under Computational Budgets in Probabilistic Decision Systems
**ICT 5143 AFGML Capstone — Problem 9** · Team: *(Student 1, roll no.)*, *(Student 2, roll no.)* · UCI Bank Marketing (`bank-additional-full.csv`)

> **Status.** All code, tests, protocol and documents are implemented. The real-data results are **not included**: the build environment had no access to the UCI server. `results/smoke/` holds a run on a **synthetic bank-like fixture** that only demonstrates file formats; it is **not evidence** and must not be quoted. Run the real pipeline below, then recompile the report/slides.

## 1. Identity and task
`Y = 1`: recorded term-deposit subscription. `D = 1`: selected for *hypothetical* pre-contact outreach (top 20 % by posterior). Decision time = before the call → `duration` and every post-contact variable are excluded (enforced by `assert_no_leakage` + tests).

## 2. Question and contribution
*Can a budget-aware fallback policy reduce the fairness degradation caused by interrupted exact inference at a fixed operation budget?* Closest method: compute-budget → group-disparity work (Das et al., ICML 2024) — conceptually; reproduced baselines are exact inference, Hardt-style post-processing after budgeted inference, and a single global fallback (`docs/related_work.md`). Contribution: a deterministic operation-count budget with evidence-dependent trips, a simulation isolating *group-dependent evidence availability* as a mechanism (with a no-dependence control), and fairness-aware fallback selection. A negative result is acceptable and will be reported as such.

## 3. Model and fairness
9-variable discrete BN (age band, job, education, default, housing, loan, pdays bin, previous bin, poutcome → y; `configs/model.yaml`); exact VE (min-fill / min-weight / min-degree) and clique-tree calibration, both implemented from scratch with factor-entry operation counters (`src/inference.py`). Primary fairness criterion: **ΔEOpp** across age band × marital status; diagnostics ΔDP, ΔEOdds, fallback-rate gaps, decision flips vs exact. Primary utility AP, allowed loss 0.01.

## 4. Results
Generated, never hand-typed: `results/tables/`, `results/figures/`, `results/metrics/` (see §6). *(Paste your real headline numbers here after the real run.)*

## 5. Installation and data
```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/acquire_data.py       # downloads bank-additional-full.csv, prints SHA-256 (record it in data/README.md)
```
If the download fails (proxy etc.), fetch the file manually from the UCI page and put it in `data/raw/`.

## 6. Reproduction
Smoke (synthetic, ≈ 1 min, proves the plumbing only):
```bash
python -m pytest tests
python scripts/run_experiments.py --mode smoke --quick --out results/demo
python scripts/evaluate.py --out results/demo --n-boot 200
python scripts/make_results.py --out results/demo --mode smoke
```
Real data (headline table first, then everything — commands mirror `docs/requirements_matrix.md`):
```bash
python scripts/prepare_data.py configs/data.yaml        # splits + group denominators
python scripts/verify_inference.py --mode real          # C-GM on random BNs
python scripts/run_experiments.py --mode real           # all experiments (~10-20 min on a laptop; 5 seeds everywhere)
python scripts/evaluate.py --out results                # paired block bootstrap, Holm, seed aggregates
python scripts/make_results.py --out results --mode real
python scripts/profile_latency.py --mode real           # wall-clock, separate from op budgets
pdflatex -interaction=nonstopmode -halt-on-error -output-directory report report/AFGML_Report.tex   # run twice
pdflatex -interaction=nonstopmode -halt-on-error -output-directory slides slides/AFGML_Presentation.tex
```
Tested: the full **smoke** pipeline, a full real-mode run (all scripts + the notebook) on a schema-identical stand-in file, and the 93-test suite on Linux/Python 3.13. **Not yet tested in the build environment:** the data download and a run on the genuine UCI file (no network access there).

**Notebook.** `notebooks/AFGML_P9_Project.ipynb` is a full driver: it locates/validates the real CSV (never falls back to synthetic data), runs the whole pipeline, and reports every Problem 9 item, including elimination traces and induced cliques (`src/diagnostics.py`), the group-unaware routing comparison (`proposed_unaware`) and a factorial synthetic study.

## 7. Evidence
`docs/` — proposal, related work, data card, model card, assumptions, protocol, requirements matrix, contributions · `report/`, `slides/`, `demo/` · `results/manifest.csv` · `notebooks/AFGML_P9_Project.ipynb`.

## 8. Contributions and attribution
See `docs/contributions.md` (students must complete it). Initial scaffold produced with an AI assistant — disclose per course policy. Data: Moro, Rita & Cortez, UCI Bank Marketing.

## Layout
`src/` inference, data, bn, fallbacks, fairness, metrics, pipeline, profiling, synth · `scripts/` entry points · `tests/` · `configs/` frozen settings · `results/` outputs.
