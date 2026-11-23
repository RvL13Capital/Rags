# Pipeline verification report

> **PIPELINE VERIFICATION - SYNTHETIC DATA.** These results exercise the code on simulated prices.
> They are not empirical market performance and say nothing about the NASDAQ-100.

Synthetic sample 2012-2026, 40 Optuna trials per fold, 10000 Monte Carlo paths, runtime 547s.

| Data set | OOS Sharpe | OOS PF | OOS max DD | OOS trades | OOS net P&L (EUR) | DSR | PBO |
|---|---:|---:|---:|---:|---:|---:|---:|
| Null (martingale) | -0.35 | 0.84 | 5.25% | 245 | -199.16 | 0.004 | 0.568 |
| Planted edge (after ablation) | 0.55 | 1.42 | 2.40% | 169 | 317.97 | 0.528 | 0.894 |

## Checks

- Null data rejected by the release gates: **PASS** (failed gates: OOS net Sharpe, Deflated Sharpe Ratio, PBO (CSCV), OOS net profit factor, OOS net P&L under 1.5x costs, Profitable walk-forward OOS folds)
- Planted edge visible out of sample (Sharpe above null and positive net P&L): **PASS**

## Planted-edge ablation

# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 111. Variants evaluated: 14.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | 0.334 | n/a | 0.144 | n/a | 1.198 | n/a | 0.38% | n/a | 2.93% | 222 |
| 1 | multi_stage_exits | - | rejected (no OOS net gain) | 0.216 | -0.118 | 0.085 | -0.059 | 1.112 | -0.086 | 0.29% | -0.09% | 2.48% | 246 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.334 | 0.000 | 0.144 | 0.000 | 1.198 | 0.000 | 0.38% | 0.00% | 2.93% | 222 |
| 3 | regime_gate | rv20 | rejected (parsimony benchmark) | 0.420 | 0.086 | 0.198 | 0.054 | 1.289 | 0.091 | 0.33% | -0.05% | 2.35% | 184 |
| 3 | regime_gate | hmm2 | RETAINED | 0.546 | 0.212 | 0.230 | 0.086 | 1.425 | 0.227 | 0.32% | -0.06% | 2.40% | 169 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | -0.095 | -0.429 | -0.031 | -0.175 | 0.913 | -0.285 | 0.25% | -0.13% | 2.24% | 78 |
| 4 | hurst_gate | - | rejected (trade starvation) | -0.116 | -0.661 | -0.030 | -0.260 | 0.668 | -0.757 | 0.00% | -0.32% | 0.21% | 2 |
| 5 | garch_vol | - | rejected (no OOS net gain) | 0.469 | -0.076 | 0.213 | -0.017 | 1.373 | -0.051 | 0.30% | -0.02% | 2.01% | 149 |
| 6 | evt_es_cap | - | rejected (no OOS net gain) | 0.328 | -0.217 | 0.073 | -0.157 | 1.269 | -0.156 | 0.19% | -0.12% | 2.54% | 159 |
| 7 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 8 | spread_gate | - | rejected (no OOS net gain) | 0.394 | -0.152 | 0.228 | -0.002 | 1.279 | -0.146 | 0.31% | -0.01% | 1.68% | 153 |
| 9 | vpin_filter | - | rejected (no OOS net gain) | 0.306 | -0.239 | 0.146 | -0.084 | 1.229 | -0.196 | 0.32% | 0.01% | 2.04% | 155 |
| 10 | event_blackout | - | rejected (no OOS net gain) | 0.448 | -0.097 | 0.163 | -0.067 | 1.345 | -0.079 | 0.32% | 0.01% | 2.79% | 165 |
| 11 | score_sizing | - | rejected (no OOS net gain) | 0.211 | -0.334 | 0.072 | -0.158 | 1.152 | -0.272 | 0.34% | 0.02% | 3.04% | 164 |
| 12 | kelly_cap | - | rejected (no OOS net gain) | -0.175 | -0.720 | -0.042 | -0.272 | 0.865 | -0.560 | 0.31% | -0.01% | 3.64% | 124 |

**Retained modules:** regime_gate (regime model: hmm2)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=938): statistic 0.808, p-value 0.2095.


Full tearsheets: `null_tearsheet.md`, `planted_tearsheet.md`.
