# Run plan, written before any result was seen

Written 2026-11-23 11:41 UTC, code at commit 59f1cbd (v7, frozen).

## Data actually used (not the planned 2012-2026 vendor exports)

- Prices: third-party public upload of NQ futures 1-minute bars (Hugging Face
  mdelcristo/NQ-F_1min_OHLCV_Parquet, MIT licence, no provenance stated),
  2015-01-02 to 2025-07-24, converted to a cash-equivalent index by
  subtracting a fair-value basis fitted per contract against the official
  daily NASDAQ-100 close from FRED.  No bid/ask history.
- Rates and FX: FRED (EFFR/SOFR splice, EURUSD).
- Events: scheduled FOMC meetings (federalreserve.gov), CPI and Employment
  Situation release dates (ALFRED).

## Fixed in advance

- Primary configuration: capital_com_eu_unleveraged_1to1 with --entry-order stop
  (the order type the broker can execute).  The other three runs are diagnostics:
  1:1 with stop_limit, leveraged with stop, leveraged with stop_limit.
- Sequential ablation with default tuning policy, 60 Optuna trials per fold
  where anything is tuned, walk-forward 36m / 12m / 6m, holdout from 2024-01-01.
- Release gates as locked in the README.  No gate, parameter or preset is
  changed after seeing results.
- Risk 0.25% per trade, EUR 5,000, lots normalised to the final close.
