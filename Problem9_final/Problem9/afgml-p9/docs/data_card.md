# Data card — UCI Bank Marketing, `bank-additional-full.csv`

- **Source/version.** UCI ML Repository dataset 222 (Moro, Rita, Cortez); file `bank-additional-full.csv`; 41,188 rows, 20 inputs + `y` (expected — verify with `data/README.md` checksum after download). Access date: *(fill)*. License: see UCI page (CC BY 4.0 at time of writing — verify).
- **Population.** Portuguese bank direct-marketing phone campaigns, ordered by contact date (no date column, no customer id). Rows are *contacts*, not necessarily distinct customers; repeated customers cannot be identified.
- **Label.** `Y = 1` iff `y == "yes"` (recorded subscription). Selection into the contacted sample is historical; conclusions do not extend to never-contacted people.
- **Decision time.** Immediately **before** the call. `duration` (known only after the call) is excluded everywhere and enforced by `assert_no_leakage` + tests. Macro-economic indicators and `month`/`day_of_week` are excluded by declared choice (`configs/data.yaml`).
- **Protected attributes.** Age (bands `<30`, `30–59`, `≥60`; primary) and marital status (secondary; audit only, not a model input). Recorded categories are limited measurements; nothing is inferred. `unknown` marital status is kept as its own group (very small → exploratory).
- **Model variables (9 + target).** age band, job, education, default, housing, loan, `pdays` bin, `previous` bin, `poutcome`. `unknown` in job/education/default/housing/loan is treated as **unobserved evidence** (`R_j = 0`).
- **Split.** Positional 60/20/20 train/val/test (date-ordered release). Bins use fixed cut points; BN fitted on train only.
- **Small groups.** Every intersection reports n, positives, negatives; < 30 relevant observations ⇒ exploratory (`results/metrics/group_denominators.csv`, `group_table.csv`).
- **Restrictions.** Raw data is not redistributed; `scripts/acquire_data.py` downloads it.
