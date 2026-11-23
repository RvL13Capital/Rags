# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 65. Variants evaluated: 14.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | -0.123 | n/a | -0.046 | n/a | 0.935 | n/a | 0.50% | n/a | 5.17% | 131 |
| 1 | multi_stage_exits | - | RETAINED | 0.046 | 0.169 | 0.015 | 0.061 | 1.016 | 0.081 | 0.36% | -0.14% | 3.77% | 137 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | 0.031 | -0.015 | 0.009 | -0.006 | 1.010 | -0.006 | 0.36% | -0.00% | 3.77% | 137 |
| 3 | regime_gate | rv20 | RETAINED | 0.289 | 0.244 | 0.150 | 0.135 | 1.138 | 0.123 | 0.35% | -0.00% | 2.59% | 113 |
| 3 | regime_gate | hmm2 | rejected (parsimony benchmark) | 0.132 | 0.087 | 0.066 | 0.051 | 1.076 | 0.060 | 0.33% | -0.03% | 2.03% | 71 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.286 | 0.241 | 0.120 | 0.105 | 1.188 | 0.172 | 0.31% | -0.04% | 2.39% | 66 |
| 4 | garch_vol | - | rejected (no OOS net gain) | 0.097 | -0.192 | 0.046 | -0.104 | 1.044 | -0.095 | 0.34% | -0.01% | 2.57% | 104 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.153 | -0.137 | 0.056 | -0.094 | 1.078 | -0.060 | 0.24% | -0.11% | 2.13% | 113 |
| 6 | spread_gate | - | SKIPPED (input data not available) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | vpin_filter | - | RETAINED | 0.343 | 0.054 | 0.187 | 0.036 | 1.168 | 0.029 | 0.35% | -0.00% | 2.46% | 112 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.341 | -0.002 | 0.185 | -0.001 | 1.167 | -0.001 | 0.35% | -0.00% | 2.46% | 112 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.353 | 0.010 | 0.194 | 0.007 | 1.179 | 0.011 | 0.30% | -0.05% | 1.97% | 112 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.231 | -0.112 | 0.124 | -0.063 | 1.129 | -0.039 | 0.33% | -0.02% | 2.10% | 80 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.000 | -0.343 | 0.000 | -0.187 | n/a | n/a | 0.00% | -0.35% | 0.00% | 0 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.000 | -0.343 | 0.000 | -0.187 | n/a | n/a | 0.00% | -0.35% | 0.00% | 0 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** multi_stage_exits, regime_gate, vpin_filter (regime model: rv20)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=412): statistic nan, p-value nan.

**Untouched holdout from 2024-01-01** (never used for a retention decision): Sharpe 0.054, profit factor 1.027, max drawdown 0.90%, trades 25, net P&L 4.45.
