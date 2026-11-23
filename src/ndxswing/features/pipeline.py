"""Builds the causal feature set consumed by the signal engine.

All statistical layers are re-estimated on rolling windows that end at the
day they are stamped with, so the same table is valid for every walk-forward
fold: nothing dated t depends on data after the close of day t.  The signal
engine additionally lags the daily table by one full session.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .garch import HORIZONS, rolling_garch_forecasts
from .hurst import rolling_hurst
from .indicators import daily_from_hourly, prepare_hourly_bars, regime_observations, trailing_pct_rank
from .regime import CausalRegimeHMM
from .vpin import compute_volume_time_vpin, vpin_percentile
from ..risk.tail import rolling_evt, rolling_tsallis_q


@dataclass(frozen=True)
class FeatureConfig:
    hmm_window: int = 756          # 36 months, the walk-forward training span
    hmm_refit_every: int = 21
    hmm_n_init: int = 4
    hurst_length: int = 1500
    hurst_order: int = 1
    hurst_boot: int = 200
    hurst_block: float = 24.0
    hurst_lcb_method: str = "normal"
    garch_window: int = 756
    garch_refit_every: int = 5
    garch_mean: str = "ar1"
    garch_vol: str = "gjr"
    evt_window: int = 504
    evt_refit_every: int = 5
    seed: int = 7
    # blocks can be switched off to keep unit tests fast
    with_regime: bool = True
    with_hurst: bool = True
    with_garch: bool = True
    with_evt: bool = True
    with_vpin: bool = True


@dataclass
class FeatureSet:
    hourly: pd.DataFrame      # prepared bars + hourly features (vpin, vpin_pct, spread_med)
    daily: pd.DataFrame       # one row per session, closed at that session
    config: FeatureConfig
    source: str = "unknown"

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.daily.index)


def _single_threaded(fn, *args):  # noqa: ANN001, ANN202 - worker shim
    """Run one estimation block with BLAS pinned to one thread (the blocks
    already run in parallel processes; nested threading only adds contention)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return fn(*args)


def _task_hmm(obs: pd.DataFrame, config: FeatureConfig, n_states: int) -> pd.DataFrame:
    if n_states == 3:
        h3 = CausalRegimeHMM(3, config.hmm_window, config.hmm_refit_every, config.hmm_n_init, seed=config.seed)
        return h3.rolling_filter(obs).drop(columns="refit_id")
    # parsimony benchmark: 2-state volatility-switching model
    h2 = CausalRegimeHMM(2, config.hmm_window, config.hmm_refit_every, config.hmm_n_init, seed=config.seed,
                         features=("r", "rv20", "vc", "gap"))
    return h2.rolling_filter(obs)[["p_stress"]].rename(columns={"p_stress": "p2_stress"})


def _task_hurst(hourly: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    return rolling_hurst(hourly, config.hurst_length, config.hurst_order, config.hurst_boot, config.hurst_block,
                         config.hurst_lcb_method, config.seed)


def _task_garch(r: pd.Series, config: FeatureConfig) -> pd.DataFrame:
    return rolling_garch_forecasts(r, config.garch_window, config.garch_refit_every, HORIZONS, config.garch_mean,
                                   config.garch_vol)


def _task_evt(r: pd.Series, config: FeatureConfig) -> pd.DataFrame:
    return rolling_evt(r, config.evt_window, config.evt_refit_every).join(
        rolling_tsallis_q(r, config.evt_window, config.evt_refit_every))


def build_feature_set(hourly_raw: pd.DataFrame, config: FeatureConfig = FeatureConfig(), source: str = "unknown",
                      verbose: bool = False, n_jobs: int = 4) -> FeatureSet:
    def log(msg: str) -> None:
        if verbose:
            print(f"[features] {msg}", flush=True)

    hourly = prepare_hourly_bars(hourly_raw)
    daily = daily_from_hourly(hourly)
    obs = regime_observations(daily)
    feat = obs.copy()
    feat["close"] = daily["close"]
    feat["sigma_rv"] = obs["rv20"] / np.sqrt(252.0)
    feat["rv20_pct"] = trailing_pct_rank(obs["rv20"], 252)

    tasks: dict[str, tuple] = {}
    if config.with_regime:
        tasks["hmm3"] = (_task_hmm, obs, config, 3)
        tasks["hmm2"] = (_task_hmm, obs, config, 2)
    if config.with_hurst:
        tasks["hurst"] = (_task_hurst, hourly[["close", "session"]], config)
    if config.with_garch:
        tasks["garch"] = (_task_garch, obs["r"], config)
    if config.with_evt:
        tasks["evt"] = (_task_evt, obs["r"], config)
    log(f"estimating {', '.join(tasks) or 'nothing'} (n_jobs={n_jobs})")
    if n_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(n_jobs, len(tasks)), mp_context=mp.get_context("fork")) as pool:
            futs = {k: pool.submit(_single_threaded, *v) for k, v in tasks.items()}
            done = {k: f.result() for k, f in futs.items()}
    else:
        done = {k: _single_threaded(*v) for k, v in tasks.items()}
    for key in ("hmm3", "hmm2", "hurst", "garch", "evt"):
        if key in done:
            feat = feat.join(done[key])

    if "spread_rel" in hourly.columns and hourly["spread_rel"].notna().any():
        hourly["spread_med"] = hourly["spread_rel"].rolling(140, min_periods=70).median()
    if config.with_vpin and (hourly["volume"] > 0).any():
        log("volume-clock VPIN")
        hourly["vpin"] = compute_volume_time_vpin(hourly)
        hourly["vpin_pct"] = vpin_percentile(hourly["vpin"])
    return FeatureSet(hourly=hourly, daily=feat, config=config, source=source)


# --------------------------------------------------------------------------- #
# Disk cache (features are expensive; key = data fingerprint + config)
# --------------------------------------------------------------------------- #


def _fingerprint(hourly_raw: pd.DataFrame, config: FeatureConfig) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(asdict(config), sort_keys=True).encode())
    h.update(str(len(hourly_raw)).encode())
    h.update(pd.util.hash_pandas_object(hourly_raw[["open", "high", "low", "close", "volume"]], index=True).values.tobytes())
    return h.hexdigest()[:16]


def load_or_build_feature_set(hourly_raw: pd.DataFrame, config: FeatureConfig, cache_dir: str | Path,
                              source: str = "unknown", verbose: bool = False) -> FeatureSet:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = _fingerprint(hourly_raw, config)
    fh, fd = cache / f"features_{key}_hourly.parquet", cache / f"features_{key}_daily.parquet"
    if fh.exists() and fd.exists():
        return FeatureSet(pd.read_parquet(fh), pd.read_parquet(fd), config, source)
    fs = build_feature_set(hourly_raw, config, source, verbose)
    fs.hourly.to_parquet(fh)
    fs.daily.to_parquet(fd)
    return fs
