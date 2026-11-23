"""Shared fixtures.  All market data used by the tests is synthetic or
hand-scripted; the suite verifies mechanics, never market performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndxswing.backtest.engine import EventDrivenBacktester
from ndxswing.config import BacktestConfig, CostModel, InstrumentSpec, ModuleFlags, StrategyParams
from ndxswing.data.synthetic import BAR_STARTS_MIN, SyntheticSpec, generate_synthetic_market
from ndxswing.features.pipeline import FeatureConfig, build_feature_set
from ndxswing.signal.engine import PreparedSignals, SignalEngine

FAST = FeatureConfig(with_regime=False, with_hurst=False, with_garch=False, with_evt=False, with_vpin=False)
ZERO_COSTS = CostModel(spread_bps=0.0, slippage_bps=0.0, commission_bps=0.0, financing_markup=0.0,
                       default_benchmark_rate=0.0)
EUR_CFD = InstrumentSpec(name="TEST-CFD", kind="cfd", quote_ccy="EUR", lot_step=0.01, min_lot=0.01)
BARS_PER_DAY = len(BAR_STARTS_MIN)


def flat_market(n_days: int = 12, price: float = 100.0, start: str = "2024-01-08") -> pd.DataFrame:
    """Hourly bars pinned at ``price`` (7 regular-session bars per weekday)."""
    days = pd.bdate_range(start, periods=n_days)
    local = [d + pd.Timedelta(minutes=m) for d in days for m in BAR_STARTS_MIN]
    ts = pd.DatetimeIndex(local).tz_localize("America/New_York").tz_convert("UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1000.0}, index=ts)


def set_bar(df: pd.DataFrame, i: int, o: float, h: float, l: float, c: float) -> None:
    df.iloc[i, df.columns.get_indexer(["open", "high", "low", "close"])] = [o, h, l, c]


class ScriptedEngine(SignalEngine):
    """Signal engine whose proposals are dictated by the test (bar -> order)."""

    def __init__(self, features, costs, script: dict[int, dict]) -> None:
        super().__init__(features, costs)
        self.script = script

    def prepare(self, params: StrategyParams, flags: ModuleFlags) -> PreparedSignals:
        n = self.n
        arr = {k: np.full(n, np.nan) for k in ("entry_stop", "entry_limit", "stop_price", "d_stop")}
        direction = np.zeros(n, dtype=np.int8)
        for i, s in self.script.items():
            direction[i] = s["direction"]
            for k in arr:
                arr[k][i] = s[k]
        one, nan = np.ones(n), np.full(n, np.nan)
        return PreparedSignals(
            params=params, flags=flags, direction=direction, atr=np.full(n, 2.0), score=one, m_reg=one,
            m_hurst=one, m_liq=one, sigma_1d=np.full(n, 0.01), es99=nan, mu_h=nan, sigma_h=nan, z_h=nan,
            spread_rel=nan, exit_long=np.zeros(n, dtype=bool), exit_short=np.zeros(n, dtype=bool),
            funnel={"signals": int((direction != 0).sum())}, **arr,
        )


LONG = dict(direction=1, entry_stop=101.0, entry_limit=101.3, stop_price=99.0, d_stop=2.0)
SHORT = dict(direction=-1, entry_stop=99.0, entry_limit=98.7, stop_price=101.0, d_stop=2.0)


def scripted_backtester(bars: pd.DataFrame, script: dict[int, dict], costs: CostModel = ZERO_COSTS,
                        **cfg_kw) -> EventDrivenBacktester:
    fs = build_feature_set(bars, FAST, source="scripted-test", n_jobs=1)
    cfg = BacktestConfig(instrument=cfg_kw.pop("instrument", EUR_CFD), costs=costs, **cfg_kw)
    return EventDrivenBacktester(ScriptedEngine(fs, costs, script), cfg)


@pytest.fixture(scope="session")
def synth_market() -> dict:
    return generate_synthetic_market(SyntheticSpec(end="2015-06-30", seed=5))


@pytest.fixture(scope="session")
def synth_features(synth_market):
    cfg = FeatureConfig(with_regime=False, with_hurst=False, with_garch=False, with_evt=False, with_vpin=True)
    return build_feature_set(synth_market["hourly"], cfg, source=synth_market["source"], n_jobs=1)
