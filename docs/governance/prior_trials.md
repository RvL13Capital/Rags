# Prior trials carried forward

Every configuration that has been scored on a data set makes the next good-looking result on
that data set less surprising. This file records what has been spent so far and how it enters
the Deflated Sharpe Ratio of later runs.

## What is spent

The 2015–2025 NASDAQ-100 run (`research_runs/2026-11-23_public_nq_2015_2025/`) scored its
variants on the stitched walk-forward OOS series from February 2019. That series is no longer
out-of-sample for any hypothesis formed after reading those results, including the long-only
and stop-limit observations in the run's README.

| Quantity | Value |
|---|---:|
| Raw trials, K_prior (4 runs × 14 ablation variants) | 56 |
| Dropped: never traded (both Hurst gates in each run, zero variance) | 8 |
| Dropped: exact duplicates of an earlier series | 0 |
| Series used | 48 |
| Average pairwise correlation of their daily returns | 0.772 |
| **Effective trials, N_eff_prior = max(1, ρ + (1 − ρ) K)** | **11.71** |
| Span of the comparison (the decision period all variants share) | 2019-02-01 to 2023-12-29, 1,236 sessions |

Modules that were skipped or blocked (spread gate, ARMA filter) produced no series and are not
counted. The four runs are two order types on two funding presets; the two presets take the same
trades, so their series correlate at about 0.81 with each other, as much as variants within one run.

Sensitivity: adding the 64 random parameter draws of each final configuration (256 series, of
which 248 are exact duplicates because the VPIN threshold rarely changes a trade) gives 56
distinct series and N_eff = 12.38. Per run, N_eff is 3.1 to 3.3.

Not counted: the post-hoc slices of the final ledgers (long against short, by year, by fill
type). They selected nothing, but any rule built from them starts from this record.

## Phase 0 on the same sample (volatility-managed passive exposure)

`research_runs/2026-11-23_phase0_volmanaged_ndx/` examined sixteen further configurations on the
spent NASDAQ-100 sample: one pre-declared primary, twelve planned sensitivities and three
thresholds of an unplanned in-or-out filter. They are not ablation variants of the breakout system
and have no N_eff computed here. If any of them is ever taken to untouched markets, the
pre-registration must add them to the 11.71 above: as a floor, one further effective trial for the
planned family and one for the unplanned filter.

## How it enters the DSR

```bash
ndxswing ablation --db data/market.duckdb --symbol NDX --prior-eff-trials 11.71 --holdout-start 2024-01-01 --out runs/abl
```

The benchmark Sharpe of the zero-edge null becomes

    N_eff_total = N_eff_cur + N_eff_prior
    V_total     = 1 / T                                               one configuration in the current run
    V_total     = (N_eff_cur · V_cur + N_eff_prior / T) / N_eff_total   several
    SR_0        = sqrt(V_total) · z(N_eff_total)

with T the number of daily returns in the tested series, V_cur the variance of the per-period
Sharpe ratios across the current run's configurations, and z(N) the expected maximum of N standard
normals. Prior trials are independent of the current family, so they carry the sampling variance of
a zero-edge Sharpe ratio (1/T) and not the narrow spread inside the current family. With
`--prior-eff-trials 0` (the default) the DSR is exactly what it was before.

## What it costs

Observed annualised Sharpe at which the DSR gate (0.95) is cleared at even odds:

| Tested series | No prior trials | 11.71 prior trials |
|---|---:|---:|
| One fixed configuration, 6.45 years | 0.65 | 1.31 |
| One fixed configuration, 10.5 years | 0.51 | 1.03 |

For a typical ablation (15 variants, correlation 0.68, cross-trial Sharpe spread 0.27 a year) on
10.5 years the benchmark SR_0 rises from 0.34 to 0.54.

The haircut is large because the linear N_eff formula treats 11.7 trials as fully independent.
That is the conservative reading.

## Rules

- A new hypothesis on NASDAQ-100 data overlapping February 2019 to July 2025 is tested with
  `--prior-eff-trials 11.71` plus whatever has been added since.
- Whether the same count applies to a different market is a judgement about how related the
  hypothesis is. The default is to carry it.
- After each further research round on the same data, add that round's N_eff here and pass the sum.

## Reproducing

```bash
python research_runs/2026-11-23_public_nq_2015_2025/prep/prior_trials.py data/market.duckdb data/feature_cache runs/prior
```

```bash
python research_runs/2026-11-23_public_nq_2015_2025/prep/prior_trials_summary.py runs/prior
```

The first re-runs the four ablations (60 trials per fold, holdout from 2024-01-01) and writes one
return series per variant; the second prints the table above. The 48 series used are stored in
`research_runs/2026-11-23_public_nq_2015_2025/data_notes/prior_variant_returns.csv.gz`. The series
were produced with the v7 code that made the original run; each final configuration reproduced its
committed net P&L (EUR 118.18, 237.74, 36.29, 172.90).
