"""Point-in-time market data store on DuckDB.

Tables
------
bars_1h        hourly bars per symbol (UTC bar start), optional quoted spread
futures_bars   RAW, unadjusted bars per individual futures contract
daily_series   named daily scalars (FX fixings, benchmark rates)
events         scheduled macro / earnings events (UTC)
ingest_log     provenance of every write

Raw contract prices are always kept; the Panama-adjusted continuous series is
derived (see :mod:`ndxswing.data.futures`) and stored as its own symbol so
that indicators use the adjusted series while notional and margin use the raw
one.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1h (
    symbol VARCHAR NOT NULL, ts TIMESTAMPTZ NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE,
    spread_rel DOUBLE, raw_close DOUBLE, source VARCHAR,
    PRIMARY KEY (symbol, ts));
CREATE TABLE IF NOT EXISTS futures_bars (
    root VARCHAR NOT NULL, contract VARCHAR NOT NULL, expiry DATE NOT NULL, ts TIMESTAMPTZ NOT NULL,
    open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE, source VARCHAR,
    PRIMARY KEY (contract, ts));
CREATE TABLE IF NOT EXISTS daily_series (
    name VARCHAR NOT NULL, date DATE NOT NULL, value DOUBLE, source VARCHAR,
    PRIMARY KEY (name, date));
CREATE TABLE IF NOT EXISTS events (
    ts TIMESTAMPTZ NOT NULL, name VARCHAR NOT NULL, tier INTEGER, source VARCHAR,
    PRIMARY KEY (ts, name));
CREATE TABLE IF NOT EXISTS ingest_log (
    ingested_at TIMESTAMPTZ DEFAULT now(), table_name VARCHAR, key VARCHAR, source VARCHAR,
    n_rows BIGINT, first_ts VARCHAR, last_ts VARCHAR);
"""


def _utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    i = pd.DatetimeIndex(idx)
    return i.tz_localize("UTC") if i.tz is None else i.tz_convert("UTC")


class MarketDataStore:
    def __init__(self, path: str | Path = "data/market.duckdb", read_only: bool = False) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(self.path, read_only=read_only)
        self.con.execute("SET TimeZone='UTC'")
        if not read_only:
            self.con.execute(SCHEMA)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "MarketDataStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _log(self, table: str, key: str, source: str, n: int, first: object, last: object) -> None:
        self.con.execute(
            "INSERT INTO ingest_log (table_name, key, source, n_rows, first_ts, last_ts) VALUES (?, ?, ?, ?, ?, ?)",
            [table, key, source, int(n), str(first), str(last)],
        )

    # ------------------------------------------------------------- hourly
    def upsert_hourly(self, symbol: str, bars: pd.DataFrame, source: str) -> int:
        """``bars``: index = bar start (tz-aware, or naive UTC), columns
        open/high/low/close/volume and optionally spread_rel, raw_close."""
        df = bars.copy()
        df.index = _utc_index(df.index)
        for col in ("spread_rel", "raw_close"):
            if col not in df:
                df[col] = float("nan")
        df = df[["open", "high", "low", "close", "volume", "spread_rel", "raw_close"]].astype(float)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        staged = df.reset_index(names="ts")
        staged.insert(0, "symbol", symbol)
        staged["source"] = source
        self.con.register("staged_bars", staged)
        self.con.execute("INSERT OR REPLACE INTO bars_1h SELECT symbol, ts, open, high, low, close, volume, "
                         "spread_rel, raw_close, source FROM staged_bars")
        self.con.unregister("staged_bars")
        if len(df):
            self._log("bars_1h", symbol, source, len(df), df.index[0], df.index[-1])
        return len(df)

    def load_hourly(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        q = "SELECT ts, open, high, low, close, volume, spread_rel, raw_close FROM bars_1h WHERE symbol = ?"
        args: list = [symbol]
        if start is not None:
            q += " AND ts >= ?"
            args.append(pd.Timestamp(start, tz="UTC").to_pydatetime())
        if end is not None:
            q += " AND ts < ?"
            args.append((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).to_pydatetime())
        df = self.con.execute(q + " ORDER BY ts", args).df()
        if df.empty:
            raise KeyError(f"no hourly bars stored for symbol {symbol!r}")
        df["ts"] = _utc_index(df["ts"])
        df = df.set_index("ts")
        return df.dropna(axis=1, how="all")

    def sources(self, symbol: str) -> list[str]:
        rows = self.con.execute("SELECT DISTINCT source FROM bars_1h WHERE symbol = ?", [symbol]).fetchall()
        return sorted(r[0] for r in rows)

    # ------------------------------------------------------- daily series
    def upsert_series(self, name: str, series: pd.Series, source: str) -> int:
        s = series.dropna()
        staged = pd.DataFrame({"name": name, "date": pd.DatetimeIndex(s.index).date, "value": s.to_numpy(float),
                               "source": source})
        self.con.register("staged_series", staged)
        self.con.execute("INSERT OR REPLACE INTO daily_series SELECT name, date, value, source FROM staged_series")
        self.con.unregister("staged_series")
        if len(s):
            self._log("daily_series", name, source, len(s), s.index[0], s.index[-1])
        return len(s)

    def load_series(self, name: str) -> pd.Series:
        df = self.con.execute("SELECT date, value FROM daily_series WHERE name = ? ORDER BY date", [name]).df()
        if df.empty:
            raise KeyError(f"no daily series stored under {name!r}")
        return pd.Series(df["value"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])), name=name)

    def has_series(self, name: str) -> bool:
        return self.con.execute("SELECT count(*) FROM daily_series WHERE name = ?", [name]).fetchall()[0][0] > 0

    # ------------------------------------------------------------- events
    def upsert_events(self, events: pd.DataFrame, source: str) -> int:
        staged = pd.DataFrame({"ts": _utc_index(events["ts"]), "name": events["name"].astype(str).to_numpy(),
                               "tier": events["tier"].astype(int).to_numpy() if "tier" in events else 1,
                               "source": source})
        self.con.register("staged_events", staged)
        self.con.execute("INSERT OR REPLACE INTO events SELECT ts, name, tier, source FROM staged_events")
        self.con.unregister("staged_events")
        if len(staged):
            self._log("events", "events", source, len(staged), staged["ts"].min(), staged["ts"].max())
        return len(staged)

    def load_events(self, max_tier: int = 1) -> pd.DataFrame:
        df = self.con.execute("SELECT ts, name, tier FROM events WHERE tier <= ? ORDER BY ts", [max_tier]).df()
        if len(df):
            df["ts"] = _utc_index(df["ts"])
        return df

    # ------------------------------------------------------------ futures
    def upsert_futures(self, root: str, bars: pd.DataFrame, source: str) -> int:
        """``bars`` columns: contract, expiry, ts, open, high, low, close, volume (RAW prices)."""
        staged = bars.copy()
        staged["ts"] = _utc_index(staged["ts"])
        staged["expiry"] = pd.to_datetime(staged["expiry"]).dt.date
        staged["root"], staged["source"] = root, source
        self.con.register("staged_fut", staged)
        self.con.execute("INSERT OR REPLACE INTO futures_bars SELECT root, contract, expiry, ts, open, high, low, "
                         "close, volume, source FROM staged_fut")
        self.con.unregister("staged_fut")
        if len(staged):
            self._log("futures_bars", root, source, len(staged), staged["ts"].min(), staged["ts"].max())
        return len(staged)

    def load_futures(self, root: str) -> pd.DataFrame:
        df = self.con.execute("SELECT contract, expiry, ts, open, high, low, close, volume FROM futures_bars "
                              "WHERE root = ? ORDER BY expiry, ts", [root]).df()
        if df.empty:
            raise KeyError(f"no futures bars stored for root {root!r}")
        df["ts"] = _utc_index(df["ts"])
        df["expiry"] = pd.to_datetime(df["expiry"])
        return df

    # ------------------------------------------------------------ summary
    def summary(self) -> pd.DataFrame:
        return self.con.execute(
            "SELECT 'bars_1h' AS tbl, symbol AS key, count(*) AS n, min(ts)::VARCHAR AS first, max(ts)::VARCHAR AS last, "
            "string_agg(DISTINCT source, ', ') AS sources FROM bars_1h GROUP BY symbol "
            "UNION ALL SELECT 'daily_series', name, count(*), min(date)::VARCHAR, max(date)::VARCHAR, "
            "string_agg(DISTINCT source, ', ') FROM daily_series GROUP BY name "
            "UNION ALL SELECT 'futures_bars', root, count(*), min(ts)::VARCHAR, max(ts)::VARCHAR, "
            "string_agg(DISTINCT source, ', ') FROM futures_bars GROUP BY root "
            "UNION ALL SELECT 'events', 'events', count(*), min(ts)::VARCHAR, max(ts)::VARCHAR, "
            "string_agg(DISTINCT source, ', ') FROM events HAVING count(*) > 0 ORDER BY 1, 2"
        ).df()
