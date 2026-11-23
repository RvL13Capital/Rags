# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 62. Variants evaluated: 14.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | 0.023 | n/a | 0.006 | n/a | 1.007 | n/a | 0.52% | n/a | 4.43% | 124 |
| 1 | multi_stage_exits | - | RETAINED | 0.198 | 0.175 | 0.073 | 0.067 | 1.084 | 0.077 | 0.33% | -0.19% | 3.59% | 129 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.183 | -0.015 | 0.067 | -0.006 | 1.077 | -0.007 | 0.33% | -0.00% | 3.59% | 129 |
| 3 | regime_gate | rv20 | RETAINED | 0.577 | 0.378 | 0.336 | 0.263 | 1.301 | 0.217 | 0.31% | -0.02% | 2.14% | 108 |
| 3 | regime_gate | hmm2 | rejected (parsimony benchmark) | 0.480 | 0.282 | 0.243 | 0.170 | 1.316 | 0.232 | 0.30% | -0.03% | 1.87% | 67 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.476 | 0.278 | 0.314 | 0.242 | 1.339 | 0.255 | 0.28% | -0.05% | 1.41% | 63 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.462 | -0.115 | 0.253 | -0.083 | 1.240 | -0.061 | 0.30% | -0.02% | 2.14% | 99 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.367 | -0.209 | 0.145 | -0.191 | 1.205 | -0.095 | 0.23% | -0.08% | 1.87% | 108 |
| 6 | spread_gate | - | SKIPPED (input data not available) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | vpin_filter | - | RETAINED | 0.633 | 0.056 | 0.383 | 0.047 | 1.337 | 0.036 | 0.31% | -0.00% | 2.05% | 107 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.628 | -0.004 | 0.380 | -0.003 | 1.335 | -0.003 | 0.31% | 0.00% | 2.05% | 107 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.674 | 0.041 | 0.411 | 0.028 | 1.371 | 0.034 | 0.26% | -0.06% | 1.61% | 107 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.596 | -0.037 | 0.326 | -0.057 | 1.314 | -0.023 | 0.31% | -0.00% | 2.24% | 107 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.000 | -0.633 | 0.000 | -0.383 | n/a | n/a | 0.00% | -0.31% | 0.00% | 0 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.000 | -0.633 | 0.000 | -0.383 | n/a | n/a | 0.00% | -0.31% | 0.00% | 0 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** multi_stage_exits, regime_gate, vpin_filter (regime model: rv20)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=412): statistic nan, p-value nan.

**Untouched holdout from 2024-01-01** (never used for a retention decision): Sharpe 0.486, profit factor 1.311, max drawdown 0.88%, trades 24, net P&L 42.22.
