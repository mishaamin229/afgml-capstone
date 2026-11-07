# Assumptions register (associational unless stated)

| # | Assumption | Justification | Checkable? | Sensitivity test | If violated |
|---|---|---|---|---|---|
| A1 | All model inputs are known before the call | field definitions of UCI file | by review + unit tests (no `duration`) | leakage tests | optimistic utility; invalid audit |
| A2 | Row order ≈ chronological contact order | UCI documentation (ordered by date) | partly (no date column) | positional split only | split not chronological |
| A3 | Fixed BN structure is an adequate *associational* model | prespecified, small | held-out log loss vs LR/GBT | sparse/dense variants (R) | worse scores; conclusions about budgets still hold for *this* model |
| A4 | `unknown` = unobserved evidence (MAR given observed) | convenience; unknown may be informative | no | with/without masking variant (extend) | biased posteriors for masked rows |
| A5 | Op count (mult + add on factor entries) is the budget currency | deterministic, hardware-independent | yes (tests) | latency measured separately | latency ≠ op count |
| A6 | Exact posterior is the reference "truth" for decisions | by definition of deviation | yes (C-GM) | — | — |
| A7 | Test rows within a date block are dependent, blocks ≈ independent | no customer id | no | block length variation | intervals too narrow/wide |
| A8 | Synthetic masks/networks represent *possible* workloads, not real ones | labelled simulation | no | rates / widths grid | do not generalise |
| A9 | No causal claim is made | associational BN | — | — | — |
