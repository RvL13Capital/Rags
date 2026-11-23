"""Reproducible ingestion into :class:`MarketDataStore`.

Sources
-------
* local CSV / Parquet files (bars, daily series, events, futures chains);
* FRED (daily FX fixings and policy rates) - needs ``FRED_API_KEY``;
* Interactive Brokers historical bars through TWS / IB Gateway (``ib_async``
  or ``ib_insync``).  NOTE: the IBKR path could not be exercised against a
  live gateway in the build environment; treat it as unverified until it has
  completed one download on your machine.

Nothing here invents data: a failed download raises, it never back-fills.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .store import MarketDataStore

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if p.suffix.lower() in (".csv", ".txt", ".gz"):
        return pd.read_csv(p)
    raise ValueError(f"unsupported file type {p.suffix!r} (use .csv or .parquet)")


def normalise_bars(df: pd.DataFrame, ts_col: str = "ts", tz: str = "UTC", bid_col: str | None = None,
                   ask_col: str | None = None) -> pd.DataFrame:
    """Rename common vendor columns, localise the timestamp (bar START) and
    derive ``spread_rel`` = (ask - bid) / mid when quotes are supplied."""
    d = df.rename(columns={c: c.strip().lower() for c in df.columns})
    alias = {"date": "ts", "datetime": "ts", "time": "ts", "timestamp": "ts", "vol": "volume"}
    d = d.rename(columns={k: v for k, v in alias.items() if k in d.columns and v not in d.columns})
    ts_col = ts_col.lower()
    if ts_col not in d:
        if isinstance(df.index, pd.DatetimeIndex):
            d[ts_col] = df.index
        else:
            raise ValueError(f"timestamp column {ts_col!r} not found; columns: {list(d.columns)}")
    ts = pd.DatetimeIndex(pd.to_datetime(d[ts_col]))
    ts = ts.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward") if ts.tz is None else ts
    d.index = ts.tz_convert("UTC").rename("ts")
    if "volume" not in d:
        d["volume"] = 0.0
    out = d[["open", "high", "low", "close", "volume"]].astype(float)
    if bid_col and ask_col:
        bid, ask = d[bid_col.lower()].astype(float), d[ask_col.lower()].astype(float)
        mid = 0.5 * (bid + ask)
        out["spread_rel"] = ((ask - bid) / mid).where((ask >= bid) & (mid > 0))
    elif "spread_rel" in d:
        out["spread_rel"] = d["spread_rel"].astype(float)
    return out.sort_index()


def ingest_bars_file(store: MarketDataStore, path: str, symbol: str, source: str, **kw: object) -> int:
    return store.upsert_hourly(symbol, normalise_bars(read_table(path), **kw), source)


def ingest_series_file(store: MarketDataStore, path: str, name: str, source: str, date_col: str = "date",
                       value_col: str = "value") -> int:
    d = read_table(path)
    s = pd.Series(pd.to_numeric(d[value_col], errors="coerce").to_numpy(), index=pd.to_datetime(d[date_col]))
    return store.upsert_series(name, s, source)


def ingest_events_file(store: MarketDataStore, path: str, source: str, tz: str = "America/New_York") -> int:
    """CSV/Parquet with columns ts, name[, tier].  Naive timestamps are read in ``tz``."""
    d = read_table(path)
    ts = pd.DatetimeIndex(pd.to_datetime(d["ts"]))
    d["ts"] = (ts.tz_localize(tz) if ts.tz is None else ts).tz_convert("UTC")
    if "tier" not in d:
        d["tier"] = 1
    return store.upsert_events(d, source)


def fetch_fred_series(series_id: str, start: str = "2010-01-01", api_key: str | None = None,
                      opener: object = urllib.request.urlopen) -> pd.Series:
    """Daily observations of a FRED series (e.g. DEXUSEU = USD per EUR, DFF)."""
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; request a free key at fred.stlouisfed.org")
    q = urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json",
                                "observation_start": start})
    with opener(f"{FRED_URL}?{q}") as resp:   # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode())
    obs = payload.get("observations")
    if not obs:
        raise RuntimeError(f"FRED returned no observations for {series_id}: {str(payload)[:200]}")
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy()
    s = pd.Series(vals, index=pd.to_datetime([o["date"] for o in obs]), name=series_id).dropna()
    return s


def ingest_fred(store: MarketDataStore, series_id: str, name: str, start: str = "2010-01-01",
                scale: float = 1.0) -> int:
    """``scale`` converts units, e.g. 0.01 for a rate quoted in percent."""
    return store.upsert_series(name, fetch_fred_series(series_id, start) * scale, f"FRED:{series_id}")


def download_ibkr_hourly(symbol: str, sec_type: str, exchange: str, currency: str = "USD", start: str = "2012-01-01",
                         what: str = "TRADES", host: str = "127.0.0.1", port: int = 7497, client_id: int = 77,
                         chunk: str = "1 M") -> pd.DataFrame:
    """Regular-session 1-hour bars from TWS / IB Gateway, paged backwards.

    ``what="BID_ASK"`` returns time-averaged bid (open) and ask (close) per
    bar, from which ``spread_rel`` can be derived and merged onto TRADES or
    MIDPOINT bars.  UNVERIFIED against a live gateway in the build environment.
    """
    try:
        from ib_async import IB, Contract, util
    except ImportError:
        try:
            from ib_insync import IB, Contract, util   # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install ib_async (pip install 'ndxswing[ibkr]') to download from IBKR") from exc
    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=True)
    try:
        contract = Contract(symbol=symbol, secType=sec_type, exchange=exchange, currency=currency)
        if not ib.qualifyContracts(contract):
            raise RuntimeError(f"IBKR could not qualify {symbol} {sec_type} {exchange}")
        end, floor = "", pd.Timestamp(start, tz="UTC")
        frames: list[pd.DataFrame] = []
        while True:
            bars = ib.reqHistoricalData(contract, endDateTime=end, durationStr=chunk, barSizeSetting="1 hour",
                                        whatToShow=what, useRTH=True, formatDate=2)
            if not bars:
                break
            df = util.df(bars)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            frames.append(df)
            first = df["date"].min()
            if first <= floor:
                break
            end = first.strftime("%Y%m%d-%H:%M:%S")
            ib.sleep(2.0)   # historical data pacing
    finally:
        ib.disconnect()
    if not frames:
        raise RuntimeError("IBKR returned no historical bars")
    allb = pd.concat(frames).drop_duplicates("date").set_index("date").sort_index()
    allb = allb[allb.index >= floor]
    allb.index = allb.index.rename("ts")
    out = allb[["open", "high", "low", "close", "volume"]].astype(float)
    out["volume"] = out["volume"].clip(lower=0.0)
    if what == "BID_ASK":
        mid = 0.5 * (allb["open"] + allb["close"])
        out["spread_rel"] = ((allb["close"] - allb["open"]) / mid).where(mid > 0)
    return out


def attach_spread(bars: pd.DataFrame, bid_ask_bars: pd.DataFrame) -> pd.DataFrame:
    """Merge ``spread_rel`` from BID_ASK bars onto price bars by timestamp."""
    out = bars.copy()
    out["spread_rel"] = bid_ask_bars["spread_rel"].reindex(out.index)
    return out


def data_quality_report(bars: pd.DataFrame) -> dict[str, object]:
    """Basic integrity checks to run after every ingest."""
    from ..features.indicators import prepare_hourly_bars

    hb = prepare_hourly_bars(bars)
    per_day = hb.groupby("session").size()
    lr = np.log(hb["close"]).diff().abs()
    return {
        "rows": int(len(hb)), "sessions": int(per_day.size), "first": str(hb.index[0]), "last": str(hb.index[-1]),
        "bars_per_session_mode": int(per_day.mode().iloc[0]),
        "short_sessions": int((per_day < per_day.mode().iloc[0]).sum()),
        "max_abs_bar_log_return": float(lr.max()),
        "bars_over_10pct_move": int((lr > 0.10).sum()),
        "zero_volume_share": float((hb["volume"] <= 0).mean()),
        "has_spread": bool("spread_rel" in hb and hb["spread_rel"].notna().any()),
    }
