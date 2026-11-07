# Demonstration (≈ 3 min)

```bash
python scripts/verify_inference.py --mode smoke          # C-GM on random BNs  -> results/smoke/metrics/cgm.json
python scripts/run_experiments.py --mode smoke --quick --out results/demo   # whole pipeline on SYNTHETIC data (~1 min)
python scripts/evaluate.py --out results/demo --n-boot 200
python scripts/make_results.py --out results/demo --mode smoke
```
Expected: C-GM `PASS`, `results/demo/figures/*.png`, `results/demo/tables/macros.tex`.
Real run: see README. **Saved fallback trace:** `results/smoke/` (synthetic; shows file formats only).
Be ready to (1) trace one VE query with an op count, (2) show where `assert_no_leakage` fires, (3) explain why a different elimination order cannot change the posterior but can change which queries trip the budget.
