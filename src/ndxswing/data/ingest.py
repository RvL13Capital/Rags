"""Reproducible ingestion into :class:`MarketDataStore`.

Sources
-------
* local CSV / Parquet files (bars, daily series, events, futures chains),
  including finer intraday bars and separate bid / ask files, which are
  merged and resampled to 09:30-anchored regular-session bars;
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
    alias = {"date": "ts", "datetime": "ts", "time": "ts", "timestamp": "ts", "ts_event": "ts", "vol": "volume"}
    d = d.rename(columns={k: v for k, v in alias.items() if k in d.columns and v not in d.columns})
    ts_col = ts_col.lower()
    if ts_col not in d:
        if isinstance(df.index, pd.DatetimeIndex):
            d[ts_col] = df.index
        else:
            raise ValueError(f"timestamp column {ts_col!r} not found; columns: {list(d.columns)}")
    raw = d[ts_col]
    if pd.api.types.is_numeric_dtype(raw):
        # epoch timestamps: the unit is read off the magnitude (seconds ~1e9 ... nanoseconds ~1e18)
        mag = float(np.nanmedian(np.abs(raw.to_numpy(dtype=float))))
        unit = "s" if mag < 1e11 else "ms" if mag < 1e14 else "us" if mag < 1e17 else "ns"
        ts = pd.DatetimeIndex(pd.to_datetime(raw.to_numpy(), unit=unit, utc=True))
    else:
        ts = pd.DatetimeIndex(pd.to_datetime(raw))
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


# --------------------------------------------------------------------------- #
# Intraday sources: bid/ask merge and 09:30-anchored session bars
# --------------------------------------------------------------------------- #

SESSION_OPEN_MIN = 9 * 60 + 30


def merge_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mid-price bars and the quoted spread from separate bid and ask bars
    (the layout CFD data vendors export).

    Mid OHLC is the average of the bid and ask OHLC, bar by bar (for high and
    low this is the usual approximation: the two extremes need not coincide).
    ``spread_rel`` = (ask close - bid close) / mid close.  Volume is the mean
    of the two sides.  Bars present on one side only are dropped and counted.
    """
    cols = ["open", "high", "low", "close"]
    j = bid[cols + ["volume"]].join(ask[cols + ["volume"]], how="inner", lsuffix="_b", rsuffix="_a")
    out = pd.DataFrame({c: 0.5 * (j[f"{c}_b"] + j[f"{c}_a"]) for c in cols}, index=j.index)
    out["volume"] = 0.5 * (j["volume_b"] + j["volume_a"])
    spread = (j["close_a"] - j["close_b"]) / out["close"]
    crossed = spread <= 0
    out["spread_rel"] = spread.where(~crossed)
    info = {"bid_rows": int(len(bid)), "ask_rows": int(len(ask)), "matched_rows": int(len(j)),
            "unmatched_rows": int(len(bid) + len(ask) - 2 * len(j)), "crossed_or_zero_spread_rows": int(crossed.sum()),
            "median_spread_bps": float(out["spread_rel"].median() * 1e4) if len(j) else float("nan")}
    return out.sort_index(), info


def suspected_scale_breaks(bars: pd.DataFrame) -> list[dict]:
    """Bars whose close differs from the previous close by more than a factor
    of three - the signature of a change in a vendor's price scaling (for
    example an index quoted with a different number of decimals in early
    history), never of a real hourly move."""
    c = bars["close"]
    ratio = (c / c.shift(1)).dropna()
    hits = ratio[(ratio > 3.0) | (ratio < 1.0 / 3.0)]
    return [{"ts": str(t), "ratio": float(r)} for t, r in hits.items()]


def rescale_before(bars: pd.DataFrame, before: str, factor: float) -> pd.DataFrame:
    """Multiply prices of all bars that start before ``before`` (UTC) by ``factor``."""
    out = bars.copy()
    m = out.index < pd.Timestamp(before, tz="UTC")
    for c in ("open", "high", "low", "close", "raw_close"):
        if c in out:
            out.loc[m, c] = out.loc[m, c] * factor
    return out


def resample_to_session_bars(bars: pd.DataFrame, calendar: object = None,
                             keep_holidays: bool = False) -> tuple[pd.DataFrame, dict]:
    """Build regular-session bars anchored at the 09:30 ET open from finer
    intraday bars (1, 5, 10, 15 or 30 minutes; index = bar start, UTC).

    Output bars start at 09:30, 10:00, 11:00, ... 15:00 New York time: a
    30-minute opening bar followed by hourly bars, seven per full session,
    which is the layout the simulator and the IBKR regular-hours feed use.

    Clock-hour bars cannot be used: the 09:00-10:00 bar straddles the open,
    so the session would start at 10:00 and lose the opening price that the
    anti-chase rule and the overnight gap are measured against.

    Only cash sessions are kept: bars on NYSE holidays and after an early
    close are dropped unless ``keep_holidays`` is set.  ``spread_rel`` is
    aggregated by its median, ``raw_close`` by its last value.
    """
    from ..features.indicators import NY
    from .sessions import REGULAR_CLOSE_MIN, nyse_calendar

    need = ["open", "high", "low", "close", "volume"]
    if any(c not in bars for c in need) or len(bars) < 2:
        raise ValueError("intraday bars need open, high, low, close, volume and at least two rows")
    df = bars[~bars.index.duplicated(keep="last")].sort_index()
    local = df.index.tz_convert(NY)
    step = pd.Series(df.index[1:] - df.index[:-1]).dt.total_seconds().div(60.0)
    freq = float(step.mode().iloc[0])
    if freq > 30 or freq <= 0 or 30 % freq != 0:
        raise ValueError(
            f"source bars are {freq:g} minutes long: a 09:30-anchored session needs bars of 1, 2, 3, 5, 6, 10, 15 or "
            "30 minutes. Clock-hour bars straddle the 09:30 open; export 1-minute or 30-minute data instead.")
    minute = pd.Series(local.hour * 60 + local.minute, index=df.index)
    if ((minute % freq) != 0).mean() > 0.01:
        raise ValueError(f"bar starts are not aligned to the {freq:g}-minute grid")
    date = pd.Series(local.tz_localize(None).normalize(), index=df.index)

    weekday = pd.Series(local.dayofweek < 5, index=df.index)
    in_day = weekday & (minute >= SESSION_OPEN_MIN) & (minute < REGULAR_CLOSE_MIN)
    dates_with_data = pd.DatetimeIndex(sorted(date[in_day].unique()))
    info: dict = {"source_minutes": freq, "source_rows": int(len(df)), "calendar": "none (holidays kept)"}
    close_min = pd.Series(REGULAR_CLOSE_MIN, index=dates_with_data)
    if not keep_holidays and len(dates_with_data):
        cal = calendar or nyse_calendar(str(dates_with_data[0].date()), str(dates_with_data[-1].date()))
        holidays = dates_with_data.difference(cal.sessions)
        close_min = cal.close_minute
        info |= {"calendar": cal.source, "holiday_dates_dropped": int(len(holidays)),
                 "holiday_dates_dropped_first": [str(d.date()) for d in holidays[:5]],
                 "sessions_without_data": int(len(cal.sessions.difference(dates_with_data))),
                 "early_close_sessions": int((cal.close_minute.reindex(dates_with_data).dropna() < REGULAR_CLOSE_MIN).sum())}
    limit = date.map(close_min)                       # NaN on a non-session date -> dropped
    keep = in_day & (minute < limit)
    df, minute, date = df[keep], minute[keep], date[keep]
    if df.empty:
        raise ValueError("no regular-session bars found (check the time zone of the timestamps)")

    bucket = minute.where(minute < 600, (minute // 60) * 60).astype(int)
    bucket[minute < 600] = SESSION_OPEN_MIN
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "spread_rel" in df:
        agg["spread_rel"] = "median"
    if "raw_close" in df:
        agg["raw_close"] = "last"
    g = df.groupby([date.rename("date"), bucket.rename("bucket")], sort=True)
    out = g.agg(agg)
    n_src = g.size()
    d_idx = out.index.get_level_values("date")
    b_idx = out.index.get_level_values("bucket").to_numpy()
    end = pd.Series(d_idx).map(close_min).to_numpy(dtype=float)
    end = (b_idx + 60 - (b_idx % 60)).clip(max=end)   # 09:30 -> 10:00, 10:00 -> 11:00, capped at the close
    expected = (end - b_idx) / freq
    out.index = (pd.DatetimeIndex(d_idx) + pd.to_timedelta(b_idx, unit="m")).tz_localize(NY).tz_convert("UTC").rename("ts")

    per_day = pd.Series(1, index=d_idx).groupby(level=0).sum()
    has_open = pd.Series(b_idx == SESSION_OPEN_MIN, index=d_idx).groupby(level=0).any()
    info |= {
        "session_rows_used": int(len(df)), "bars": int(len(out)), "sessions": int(per_day.size),
        "first": str(out.index[0]), "last": str(out.index[-1]),
        "bars_per_session_mode": int(per_day.mode().iloc[0]),
        "sessions_missing_open_bar": int((~has_open).sum()),
        "incomplete_bars": int((n_src.to_numpy() < expected - 1e-9).sum()),
        "incomplete_bar_share": float((n_src.to_numpy() < expected - 1e-9).mean()),
    }
    return out, info


def load_intraday(path: str | None = None, bid_path: str | None = None, ask_path: str | None = None,
                  ts_col: str = "ts", tz: str = "UTC", bid_col: str | None = None,
                  ask_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    """One price file (optionally with bid/ask columns) or a bid file plus an ask file."""
    if (path is None) == (bid_path is None or ask_path is None):
        raise ValueError("pass either --file, or both --bid-file and --ask-file")
    if path is not None:
        return normalise_bars(read_table(path), ts_col=ts_col, tz=tz, bid_col=bid_col, ask_col=ask_col), {}
    bid = normalise_bars(read_table(bid_path), ts_col=ts_col, tz=tz)
    ask = normalise_bars(read_table(ask_path), ts_col=ts_col, tz=tz)
    return merge_bid_ask(bid, ask)


def ingest_intraday(store: MarketDataStore, symbol: str, source: str, path: str | None = None,
                    bid_path: str | None = None, ask_path: str | None = None, ts_col: str = "ts", tz: str = "UTC",
                    bid_col: str | None = None, ask_col: str | None = None, keep_holidays: bool = False,
                    scale_before: str | None = None, scale_factor: float = 1.0) -> dict:
    """Finer intraday bars -> merged quotes -> optional rescale -> 09:30-anchored
    session bars -> store.  Returns a report of everything that was dropped."""
    fine, merge_info = load_intraday(path, bid_path, ask_path, ts_col, tz, bid_col, ask_col)
    if scale_before is not None:
        fine = rescale_before(fine, scale_before, scale_factor)
    bars, info = resample_to_session_bars(fine, keep_holidays=keep_holidays)
    store.upsert_hourly(symbol, bars, f"{source} (resampled from {info['source_minutes']:g}-minute bars)")
    breaks = suspected_scale_breaks(bars)
    report = {"resample": info, "suspected_scale_breaks": breaks[:10], "n_suspected_scale_breaks": len(breaks)}
    if merge_info:
        report["bid_ask_merge"] = merge_info
    if scale_before is not None:
        report["rescaled"] = {"before": scale_before, "factor": scale_factor}
    return report


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
