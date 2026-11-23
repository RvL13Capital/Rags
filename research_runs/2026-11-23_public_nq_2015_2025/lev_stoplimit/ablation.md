# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 62. Variants evaluated: 14.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | -0.131 | n/a | -0.054 | n/a | 0.930 | n/a | 0.55% | n/a | 4.48% | 124 |
| 1 | multi_stage_exits | - | RETAINED | 0.040 | 0.171 | 0.012 | 0.067 | 1.014 | 0.084 | 0.33% | -0.21% | 3.67% | 129 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.023 | -0.017 | 0.006 | -0.006 | 1.007 | -0.007 | 0.33% | -0.00% | 3.70% | 129 |
| 3 | regime_gate | rv20 | RETAINED | 0.415 | 0.375 | 0.234 | 0.222 | 1.209 | 0.194 | 0.32% | -0.02% | 2.19% | 108 |
| 3 | regime_gate | hmm2 | rejected (parsimony benchmark) | 0.316 | 0.276 | 0.153 | 0.140 | 1.198 | 0.184 | 0.30% | -0.03% | 1.94% | 67 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.372 | 0.332 | 0.241 | 0.229 | 1.259 | 0.245 | 0.28% | -0.05% | 1.42% | 63 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.313 | -0.102 | 0.165 | -0.069 | 1.157 | -0.052 | 0.30% | -0.01% | 2.20% | 99 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.232 | -0.183 | 0.089 | -0.146 | 1.126 | -0.083 | 0.23% | -0.08% | 1.92% | 108 |
| 6 | spread_gate | - | SKIPPED (input data not available) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | vpin_filter | - | RETAINED | 0.472 | 0.057 | 0.279 | 0.044 | 1.243 | 0.034 | 0.31% | -0.00% | 2.08% | 107 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.470 | -0.002 | 0.277 | -0.001 | 1.242 | -0.001 | 0.31% | 0.00% | 2.08% | 107 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.502 | 0.029 | 0.292 | 0.014 | 1.266 | 0.023 | 0.26% | -0.05% | 1.67% | 107 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.351 | -0.121 | 0.184 | -0.095 | 1.181 | -0.062 | 0.31% | -0.00% | 2.20% | 105 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.000 | -0.472 | 0.000 | -0.279 | n/a | n/a | 0.00% | -0.31% | 0.00% | 0 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.000 | -0.472 | 0.000 | -0.279 | n/a | n/a | 0.00% | -0.31% | 0.00% | 0 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** multi_stage_exits, regime_gate, vpin_filter (regime model: rv20)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=412): statistic nan, p-value nan.

**Untouched holdout from 2024-01-01** (never used for a retention decision): Sharpe 0.332, profit factor 1.200, max drawdown 1.02%, trades 25, net P&L 28.92.
