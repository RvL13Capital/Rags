"""Reproducible ingestion into :class:`MarketDataStore`.

Sources
-------
* local CSV / Parquet files (bars, daily series, events, futures chains);
* FRED (daily FX fixings and overnight rates, with USD_RATE spliced from EFFR
  into SOFR on 2018-04-03): the JSON API when
  ``FRED_API_KEY`` is set, otherwise the public keyless CSV download
  (``fred.stlouisfed.org/graph/fredgraph.csv?id=EFFR``);
* Interactive Brokers historical bars through TWS / IB Gateway (``ib_async``
  or ``ib_insync``).  NOTE: the IBKR path could not be exercised against a
  live gateway in the build environment; treat it as unverified until it has
  completed one download on your machine.

Nothing here invents data: a failed download raises, it never back-fills.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .store import MarketDataStore

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


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
                      opener: object = None) -> pd.Series:
    """Daily observations of a FRED series (e.g. DEXUSEU = USD per EUR, DFF)."""
    key = api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; request a free key at fred.stlouisfed.org")
    q = urllib.parse.urlencode({"series_id": series_id, "api_key": key, "file_type": "json",
                                "observation_start": start})
    opener = opener or urllib.request.urlopen
    with opener(f"{FRED_URL}?{q}") as resp:   # noqa: S310 - fixed https host
        payload = json.loads(resp.read().decode())
    obs = payload.get("observations")
    if not obs:
        raise RuntimeError(f"FRED returned no observations for {series_id}: {str(payload)[:200]}")
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs]), errors="coerce").to_numpy()
    s = pd.Series(vals, index=pd.to_datetime([o["date"] for o in obs]), name=series_id).dropna()
    return s


def fetch_fred_csv(series_id: str, start: str = "2010-01-01", opener: object = None) -> pd.Series:
    """Daily observations from FRED's public CSV download - no API key."""
    # No custom User-Agent: FRED's front end silently drops requests whose
    # agent string it does not recognise, while the urllib default is served.
    url = f"{FRED_CSV_URL}?{urllib.parse.urlencode({'id': series_id, 'cosd': start})}"
    opener = opener or urllib.request.urlopen
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with opener(url, timeout=60) as resp:   # noqa: S310 - fixed https host
                text = resp.read().decode("utf-8", errors="replace")
            break
        except (OSError, http.client.HTTPException) as exc:   # timeouts, resets, dropped connections
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"FRED CSV download for {series_id} failed after 3 attempts: {last_exc}") from last_exc
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    header = [h.strip() for h in lines[0].split(",")] if lines else []
    if len(header) != 2 or header[0] not in ("observation_date", "DATE") or header[1] != series_id:
        raise RuntimeError(f"FRED CSV for {series_id} has an unexpected layout: {text[:120]!r}")
    dates, vals = [], []
    for ln in lines[1:]:
        d, _, v = ln.partition(",")
        dates.append(d.strip())
        vals.append(v.strip())
    s = pd.Series(pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(), index=pd.to_datetime(dates),
                  name=series_id).dropna()     # FRED marks holidays with "." or an empty field
    if s.empty:
        raise RuntimeError(f"FRED CSV returned no observations for {series_id} from {start}")
    return s


def fetch_fred(series_id: str, start: str = "2010-01-01", api_key: str | None = None) -> tuple[pd.Series, str]:
    """(series, source label): JSON API with a key, keyless CSV without one."""
    if api_key or os.environ.get("FRED_API_KEY"):
        return fetch_fred_series(series_id, start, api_key), f"FRED:{series_id} (API)"
    return fetch_fred_csv(series_id, start), f"FRED:{series_id} (public CSV)"


def ingest_fred(store: MarketDataStore, series_id: str, name: str, start: str = "2010-01-01",
                scale: float = 1.0) -> int:
    """``scale`` converts units, e.g. 0.01 for a rate quoted in percent."""
    s, source = fetch_fred(series_id, start)
    return store.upsert_series(name, s * scale, source)


SOFR_START = pd.Timestamp("2018-04-03")   # first SOFR publication


def splice_usd_rate(effr: pd.Series, sofr: pd.Series, switch: pd.Timestamp = SOFR_START) -> pd.Series:
    """USD overnight benchmark for CFD funding: EFFR before ``switch``, SOFR
    from ``switch`` onward, falling back to EFFR on any date SOFR is missing."""
    idx = effr.index.union(sofr.index)
    e, s_ = effr.reindex(idx), sofr.reindex(idx)
    out = e.where(idx < switch, s_.where(s_.notna(), e))
    return out.dropna().rename("USD_RATE")


def ingest_usd_macro(store: MarketDataStore, start: str = "2010-01-01") -> dict[str, dict]:
    """Everything the backtest needs besides price bars and the event
    calendar, in one call and without a key:

    EFFR      effective federal funds rate, decimal p.a.
    SOFR      secured overnight financing rate, decimal p.a. (from 2018-04-03)
    USD_RATE  the benchmark used for CFD funding: EFFR before 2018-04-03,
              SOFR from then on, EFFR on any date SOFR is missing
    EURUSD    USD per EUR, the daily noon fixing (DEXUSEU)
    """
    out: dict[str, dict] = {}

    def put(name: str, s: pd.Series, source: str) -> None:
        store.upsert_series(name, s, source)
        out[name] = {"rows": int(len(s)), "first": str(s.index[0].date()), "last": str(s.index[-1].date()),
                     "last_value": float(s.iloc[-1]), "source": source}

    effr, src_e = fetch_fred("EFFR", start)
    sofr, src_s = fetch_fred("SOFR", start)
    fx, src_x = fetch_fred("DEXUSEU", start)
    effr, sofr = effr * 0.01, sofr * 0.01
    put("EFFR", effr, src_e)
    put("SOFR", sofr, src_s)
    usd = splice_usd_rate(effr, sofr)
    put("USD_RATE", usd, f"splice: {src_e} before {SOFR_START.date()}, {src_s} after (EFFR where SOFR is missing)")
    out["USD_RATE"]["sofr_days"] = int((usd.index >= SOFR_START).sum())
    out["USD_RATE"]["effr_fallback_days"] = int(((usd.index >= SOFR_START) & ~usd.index.isin(sofr.index)).sum())
    put("EURUSD", fx, src_x)
    return out


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
