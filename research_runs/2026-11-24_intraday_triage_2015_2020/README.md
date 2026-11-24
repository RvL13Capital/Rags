# Intraday triage on 2015–2020

Two rounds of quick tests of intraday rules that are flat before the close, run on 2015–2020 only.
Bars from 2021 onwards were filtered out at load in every script here. Nothing was tuned; each rule
was run once as written. The scripts in `as_run/` are kept exactly as they were executed.

| Round | Script | Rules | Outcome |
|---|---|---|---|
| 1, manual fixed-time brackets | `as_run/triage.py` | opening-range breakout; failed breakout (two targets); gap fill; power-hour continuation | none with a net edge; gross expectancy about zero except the gap fill (58 trades) |
| 2, automated bots | `as_run/bots.py`, `as_run/verify_a.py` | A noise-area momentum (symmetric and trend-aligned); B squeeze and volume expansion; C jump-filtered VWAP limit fade | A positive under basis-point costs (net +0.086R, t = 2.74); B negative; C about two trades a year |

Round 1 charged costs in fixed index points (1.0 + 0.5 a leg). Round 2 charged them in basis points
of the contemporaneous price. Under index-point costs Bot A's 2015–2020 result is +0.022R
(t = 0.70).

B and C are archived. Symmetric Bot A was frozen in `src/ndxswing/research/intraday.py` and taken to
a single out-of-sample test: [`../2026-11-24_bot_a_oos_2021_2025/`](../2026-11-24_bot_a_oos_2021_2025/README.md).

`results/` holds the two scorecards and the trade ledgers used later (all four Bot A symmetric
ledgers, and one ledger each for the trend-aligned variant, B and C). The nine rules count as 8.73
effective trials (average pairwise correlation of their daily results 0.034).
