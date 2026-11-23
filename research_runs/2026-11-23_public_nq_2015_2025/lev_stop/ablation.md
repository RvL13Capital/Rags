# Sequential ablation (stitched walk-forward OOS)

Retention floor on trade count: 65. Variants evaluated: 14.

| Step | Module | Variant | Status | OOS Sharpe | d Sharpe | Calmar | d Calmar | PF | d PF | ES99 | d ES99 | Max DD | Trades |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | baseline | - | BASELINE | -0.310 | n/a | -0.095 | n/a | 0.849 | n/a | 0.50% | n/a | 5.80% | 131 |
| 1 | multi_stage_exits | - | RETAINED | -0.116 | 0.194 | -0.046 | 0.049 | 0.950 | 0.100 | 0.36% | -0.14% | 3.95% | 137 |
| 2 | trend_flip_exit | - | rejected (no OOS net gain) | -0.131 | -0.015 | -0.051 | -0.005 | 0.944 | -0.006 | 0.36% | 0.00% | 3.96% | 137 |
| 3 | regime_gate | rv20 | RETAINED | 0.122 | 0.238 | 0.056 | 0.102 | 1.055 | 0.105 | 0.36% | -0.00% | 2.84% | 113 |
| 3 | regime_gate | hmm2 | rejected (parsimony benchmark) | -0.030 | 0.086 | -0.017 | 0.029 | 0.980 | 0.030 | 0.33% | -0.03% | 2.21% | 71 |
| 3 | regime_gate | hmm3 | rejected (parsimony benchmark) | 0.147 | 0.263 | 0.057 | 0.103 | 1.091 | 0.141 | 0.32% | -0.04% | 2.55% | 67 |
| 4 | garch_vol | - | rejected (no OOS net gain) | -0.058 | -0.180 | -0.030 | -0.085 | 0.970 | -0.085 | 0.35% | -0.01% | 2.83% | 104 |
| 5 | evt_es_cap | - | rejected (no OOS net gain) | 0.009 | -0.113 | 0.002 | -0.054 | 1.003 | -0.052 | 0.25% | -0.11% | 2.19% | 113 |
| 6 | spread_gate | - | SKIPPED (input data not available) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| 7 | vpin_filter | - | RETAINED | 0.177 | 0.055 | 0.087 | 0.031 | 1.082 | 0.028 | 0.35% | -0.00% | 2.68% | 112 |
| 8 | event_blackout | - | rejected (no OOS net gain) | 0.176 | -0.001 | 0.087 | -0.000 | 1.082 | -0.000 | 0.35% | -0.00% | 2.68% | 112 |
| 9 | score_sizing | - | rejected (no OOS net gain) | 0.176 | -0.001 | 0.087 | 0.000 | 1.085 | 0.003 | 0.31% | -0.05% | 2.15% | 112 |
| 10 | kelly_cap | - | rejected (no OOS net gain) | 0.081 | -0.096 | 0.038 | -0.049 | 1.044 | -0.039 | 0.33% | -0.03% | 2.17% | 76 |
| 11 | hurst_gate | lcb | rejected (trade starvation) | 0.000 | -0.177 | 0.000 | -0.087 | n/a | n/a | 0.00% | -0.35% | 0.00% | 0 |
| 11 | hurst_gate | point | rejected (trade starvation) | 0.000 | -0.177 | 0.000 | -0.087 | n/a | n/a | 0.00% | -0.35% | 0.00% | 0 |
| 12 | arma_z_filter | - | BLOCKED (Pesaran-Timmermann not significant) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

**Retained modules:** multi_stage_exits, regime_gate, vpin_filter (regime model: rv20)

Pesaran-Timmermann test of the ARMA forecast sign (H=3d, n=412): statistic nan, p-value nan.

**Untouched holdout from 2024-01-01** (never used for a retention decision): Sharpe -0.222, profit factor 0.879, max drawdown 1.19%, trades 25, net P&L -21.04.
