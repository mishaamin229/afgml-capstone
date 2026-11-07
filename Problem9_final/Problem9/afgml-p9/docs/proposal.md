# Proposal — ICT 5143, Problem 9

**Title.** Fairness Preservation under Computational Budgets in Probabilistic Decision Systems (UCI Bank Marketing).

## Research question
Can a budget-aware fallback policy reduce the fairness degradation caused by interrupted exact inference in a Bayesian-network outreach model, at a fixed deterministic operation budget?

## Decision context
`D = 1`: customer selected for *hypothetical* pre-contact outreach (capacity-limited). `Y = 1`: recorded term-deposit subscription. Stakeholders: customers, outreach teams, serving engineers, auditors. No causal marketing claims.

## One falsifiable contribution sentence
> At a fixed operation budget, choosing the fallback by validation worst-group decision deviation from exact inference (subject to ≤ 0.01 absolute AP loss) yields a smaller ΔEOpp than a single global fallback, utility-only fallback selection, and ordinary threshold adjustment applied after budgeted inference, on the natural evidence workload.

**Refuted if** the Holm-adjusted paired 95 % interval for ΔEOpp(proposed) − ΔEOpp(comparator) contains 0 or is positive for any family member, or if the proposed rule exceeds the AP-loss tolerance on test. A null/negative outcome is a reportable finding.

**Mechanism claim (simulation).** Group-dependent *evidence availability* alone — with exact inference unchanged — produces group differences in budget-triggered fallback rates; a workload with identical availability across groups does not. Exact elimination order never changes the posterior; orders only change which queries exceed the budget.

## Preregistered choices (frozen in `configs/`)
Primary predictive metric AP; allowed AP loss 0.01; primary fairness ΔEOpp over age band × marital; capacity 20 %; primary budget = 75th percentile of natural *validation* exact op counts; confirmatory family `{global_prior, util_sel, thr_adj_post_budget}` with Holm; synthetic-mask workloads are exploratory; seeds 0–4.

## Amendments log
| Date | Change | Type |
|---|---|---|
| *(fill)* | Configs frozen before any real-data test outcome was inspected — **confirm this is still true before your first real run** | prior choice |
| *(fill)* | e.g. macro/calendar columns excluded; `unknown` → unobserved | prior / retrospective (state which) |
