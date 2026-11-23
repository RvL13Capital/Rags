"""09:30-anchored session bars from finer intraday data, bid/ask merge, the
NYSE session calendar and the `ingest intraday` command.  All inputs are
hand-built; nothing here is market data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndxswing.cli import main
from ndxswing.data import ingest as ing
from ndxswing.data.sessions import nyse_calendar, rule_based_calendar
from ndxswing.data.store import MarketDataStore
from ndxswing.features.indicators import prepare_hourly_bars

NY = "America/New_York"
SESSION_STARTS = ["09:30", "10:00", "11:00", "12:00", "13:00", "14:00", "15:00"]


def minute_bars(start: str, end: str, freq: str = "1min", base: float = 18_000.0) -> pd.DataFrame:
    """Round-the-clock bars (weekends included, as a 24h quote feed delivers
    them) on a deterministic ramp: bar k opens at base + k and closes one higher."""
    ts = pd.date_range(start, end, freq=freq, tz="UTC", inclusive="left")
    k = np.arange(len(ts), dtype=float)
    return pd.DataFrame({"open": base + k, "high": base + k + 1.5, "low": base + k - 0.5, "close": base + k + 1.0,
                         "volume": 2.0}, index=ts.rename("ts"))


def et(bars: pd.DataFrame) -> pd.DatetimeIndex:
    return bars.index.tz_convert(NY)


# ------------------------------------------------------------------ resampling


def test_session_bars_are_anchored_at_the_0930_open():
    fine = minute_bars("2024-03-04", "2024-03-16")                # spans the US daylight-saving switch on 10 March
    bars, info = ing.resample_to_session_bars(fine)
    local = et(bars)
    by_day = pd.Series(local.strftime("%H:%M"), index=local.normalize().tz_localize(None)).groupby(level=0).agg(list)
    assert len(by_day) == 10 and all(v == SESSION_STARTS for v in by_day)
    assert info["bars_per_session_mode"] == 7 and info["sessions"] == 10 and info["source_minutes"] == 1.0
    assert info["sessions_missing_open_bar"] == 0 and info["incomplete_bars"] == 0
    # winter time on Friday 8 March (14:30 UTC), summer time on Monday 11 March (13:30 UTC)
    assert bars.index[bars.index.normalize() == pd.Timestamp("2024-03-08", tz="UTC")][0].strftime("%H:%M") == "14:30"
    assert bars.index[bars.index.normalize() == pd.Timestamp("2024-03-11", tz="UTC")][0].strftime("%H:%M") == "13:30"

    day = pd.Timestamp("2024-03-06")
    src = fine[(et(fine) >= day.tz_localize(NY) + pd.Timedelta("9h30m")) & (et(fine) < day.tz_localize(NY) + pd.Timedelta("16h"))]
    got = bars[et(bars).normalize().tz_localize(None) == day]
    first, last = got.iloc[0], got.iloc[-1]
    open_0930 = src.iloc[0]
    assert first["open"] == open_0930["open"]                     # the true session open, not a 09:00 or 10:00 print
    assert first["close"] == src.iloc[29]["close"] and first["volume"] == 30 * 2.0
    assert first["high"] == src.iloc[:30]["high"].max() and first["low"] == src.iloc[:30]["low"].min()
    assert got.iloc[1]["open"] == src.iloc[30]["open"] and got.iloc[1]["volume"] == 60 * 2.0
    assert last["close"] == src.iloc[-1]["close"]                 # the 15:59 minute closes the session
    assert got["volume"].sum() == len(src) * 2.0                  # nothing inside the session lost or double counted

    # the result is exactly what the feature pipeline expects
    hb = prepare_hourly_bars(bars)
    assert len(hb) == len(bars) and hb.groupby("session").size().eq(7).all()
    assert ing.data_quality_report(bars)["bars_per_session_mode"] == 7


def test_any_grid_that_divides_thirty_minutes_gives_the_same_bars():
    fine = minute_bars("2024-05-06", "2024-05-11")
    ref, _ = ing.resample_to_session_bars(fine)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for freq in ("5min", "15min", "30min"):
        coarse = fine.resample(freq).agg(agg).dropna()
        got, info = ing.resample_to_session_bars(coarse)
        assert info["source_minutes"] == float(freq.replace("min", ""))
        pd.testing.assert_frame_equal(got, ref)


def test_clock_hour_and_misaligned_bars_are_refused():
    fine = minute_bars("2024-05-06", "2024-05-11")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    hourly = fine.resample("60min").agg(agg).dropna()
    with pytest.raises(ValueError, match="straddle the 09:30 open"):
        ing.resample_to_session_bars(hourly)
    # ... because this is what the session filter makes of clock-hour bars: no 09:30 bar, six bars a day
    lost = prepare_hourly_bars(hourly)
    assert lost.groupby("session").size().eq(6).all() and et(lost)[0].strftime("%H:%M") == "10:00"
    with pytest.raises(ValueError, match="60 minutes|20 minutes"):
        ing.resample_to_session_bars(fine.resample("20min").agg(agg).dropna())
    shifted = fine.resample("30min", offset="7min").agg(agg).dropna()
    with pytest.raises(ValueError, match="not aligned"):
        ing.resample_to_session_bars(shifted)
    with pytest.raises(ValueError, match="no regular-session bars"):
        ing.resample_to_session_bars(fine[et(fine).hour < 6])


def test_gaps_in_the_source_are_reported_not_hidden():
    fine = minute_bars("2024-05-06", "2024-05-11")
    local = et(fine)
    hole = (local.normalize() == pd.Timestamp("2024-05-08", tz=NY)) & (local.hour == 9)      # no open half hour
    thin = (local.normalize() == pd.Timestamp("2024-05-09", tz=NY)) & (local.hour == 11) & (local.minute >= 20)
    bars, info = ing.resample_to_session_bars(fine[~hole & ~thin])
    assert info["sessions_missing_open_bar"] == 1 and info["incomplete_bars"] == 1
    assert len(bars) == 5 * 7 - 1
    day = bars[et(bars).normalize() == pd.Timestamp("2024-05-08", tz=NY)]
    assert et(day)[0].strftime("%H:%M") == "10:00"                 # kept, and flagged, rather than back-filled


# --------------------------------------------------------- holidays, early closes


@pytest.mark.parametrize("prefer_package", [True, False])
def test_holidays_and_early_closes_are_not_sessions(prefer_package):
    fine = minute_bars("2024-11-25", "2024-12-03", freq="5min")   # Thanksgiving week: a 24h feed quotes throughout
    cal = nyse_calendar("2024-11-25", "2024-12-02", prefer_package=prefer_package)
    bars, info = ing.resample_to_session_bars(fine, calendar=cal)
    days = pd.Series(1, index=et(bars).normalize().tz_localize(None)).groupby(level=0).sum()
    assert [str(d.date()) for d in days.index] == ["2024-11-25", "2024-11-26", "2024-11-27", "2024-11-29", "2024-12-02"]
    assert days.loc["2024-11-29"] == 4                            # 13:00 close: 09:30, 10:00, 11:00, 12:00
    friday = bars[et(bars).normalize() == pd.Timestamp("2024-11-29", tz=NY)]
    assert et(friday)[-1].strftime("%H:%M") == "12:00" and friday["volume"].iloc[-1] == 12 * 2.0
    assert info["holiday_dates_dropped"] == 1 and info["holiday_dates_dropped_first"] == ["2024-11-28"]
    assert info["early_close_sessions"] == 1 and info["incomplete_bars"] == 0

    raw, info_raw = ing.resample_to_session_bars(fine, keep_holidays=True)
    assert len(raw) == 6 * 7 and info_raw["calendar"].startswith("none")


def test_rule_based_calendar_known_dates():
    cal = rule_based_calendar("2012-01-01", "2026-11-20")
    s = cal.sessions
    for closed in ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09",        # Sandy, days of mourning
                   "2024-03-29", "2022-06-20", "2023-01-02", "2021-07-05", "2022-12-26"):
        assert pd.Timestamp(closed) not in s, closed
    for open_ in ("2021-06-18",        # Juneteenth became a market holiday in 2022
                  "2021-12-31",        # a Saturday New Year's Day is not observed on the Friday
                  "2024-11-29", "2024-12-24"):
        assert pd.Timestamp(open_) in s, open_
    assert cal.close_minute.loc["2024-11-29"] == 780 and cal.close_minute.loc["2024-12-24"] == 780
    assert cal.close_minute.loc["2023-07-03"] == 780 and cal.close_minute.loc["2015-07-02"] == 960
    assert cal.close_minute.loc["2024-11-27"] == 960


def test_rule_based_calendar_matches_the_maintained_one():
    pytest.importorskip("exchange_calendars")
    ref = nyse_calendar("2012-01-01", "2026-11-20")
    mine = rule_based_calendar("2012-01-01", "2026-11-20")
    assert ref.source.startswith("exchange_calendars") and len(ref.sessions) > 3700
    assert ref.sessions.equals(mine.sessions)
    assert (ref.close_minute == mine.close_minute).all()
    assert int((ref.close_minute < 960).sum()) >= 30


# -------------------------------------------------------------- bid / ask merge


def test_bid_ask_merge():
    fine = minute_bars("2024-05-06 13:00", "2024-05-06 15:00")
    bid, ask = fine - 0.6, fine + 0.6
    bid["volume"], ask["volume"] = 4.0, 6.0
    ask = ask.iloc[:-5]                                           # the ask file ends early
    ask.iloc[10, ask.columns.get_loc("close")] = bid["close"].iloc[10] - 0.1      # one crossed quote
    mid, info = ing.merge_bid_ask(bid, ask)
    assert len(mid) == len(fine) - 5 and info["unmatched_rows"] == 5 and info["matched_rows"] == len(mid)
    assert np.allclose(mid["open"], fine["open"].iloc[:-5]) and np.allclose(mid["high"], fine["high"].iloc[:-5])
    assert (mid["volume"] == 5.0).all()
    ok = mid["spread_rel"].dropna()
    assert len(ok) == len(mid) - 1 and info["crossed_or_zero_spread_rows"] == 1
    assert ok.iloc[0] == pytest.approx(1.2 / fine["close"].iloc[0])
    assert info["median_spread_bps"] == pytest.approx(1.2 / 18_060.0 * 1e4, rel=0.01)


def test_spread_survives_resampling_as_the_bucket_median():
    fine = minute_bars("2024-05-06", "2024-05-08")
    fine["spread_rel"] = 1e-4
    local = et(fine)
    spike = (local.normalize() == pd.Timestamp("2024-05-06", tz=NY)) & (local.hour == 9) & (local.minute == 31)
    fine.loc[spike, "spread_rel"] = 50e-4                          # one wild quote must not set the bar's spread
    bars, _ = ing.resample_to_session_bars(fine)
    assert np.allclose(bars["spread_rel"], 1e-4)


# ------------------------------------------------------- timestamps and scaling


def test_epoch_timestamps_and_vendor_column_names():
    base = pd.Timestamp("2024-05-06 13:30", tz="UTC")
    for unit, div in (("s", 10**9), ("ms", 10**6), ("us", 10**3), ("ns", 1)):
        raw = pd.DataFrame({"timestamp": [base.value // div, (base + pd.Timedelta("1min")).value // div],
                            "Open": [1.0, 2.0], "High": [2.0, 3.0], "Low": [0.5, 1.5], "Close": [1.5, 2.5]})
        out = ing.normalise_bars(raw)
        assert out.index[0] == base and out.index[1] == base + pd.Timedelta("1min"), unit
        assert (out["volume"] == 0.0).all()
    named = pd.DataFrame({"ts_event": ["2024-05-06T13:30:00Z"], "open": [1.0], "high": [2.0], "low": [0.5],
                          "close": [1.5], "volume": [7]})
    assert ing.normalise_bars(named).index[0] == base


def test_scale_break_detection_and_repair():
    fine = minute_bars("2024-05-06", "2024-05-11", freq="30min")
    cut = pd.Timestamp("2024-05-08", tz="UTC")
    broken = fine.copy()
    broken.loc[broken.index < cut, ["open", "high", "low", "close"]] *= 100.0     # early history quoted 100x
    bars, _ = ing.resample_to_session_bars(broken)
    hits = ing.suspected_scale_breaks(bars)
    assert len(hits) == 1 and hits[0]["ratio"] == pytest.approx(0.01, rel=0.05)
    assert hits[0]["ts"].startswith("2024-05-08")
    fixed, _ = ing.resample_to_session_bars(ing.rescale_before(broken, "2024-05-08", 0.01))
    assert ing.suspected_scale_breaks(fixed) == []
    pd.testing.assert_frame_equal(fixed, ing.resample_to_session_bars(fine)[0])


# ------------------------------------------------------------------------- CLI


def test_ingest_intraday_command_end_to_end(tmp_path, capsys):
    fine = minute_bars("2024-11-18", "2024-12-03")
    for side, shift in (("bid", -0.9), ("ask", +0.9)):
        f = (fine[["open", "high", "low", "close"]] + shift).assign(volume=fine["volume"])
        f = f.reset_index().rename(columns={"ts": "timestamp"})
        f["timestamp"] = f["timestamp"].astype("int64") // (10**6 if str(f["timestamp"].dtype).endswith("ns, UTC]") else 10**3)
        f.to_parquet(tmp_path / f"{side}.parquet")
    db = str(tmp_path / "m.duckdb")
    main(["ingest", "intraday", "--db", db, "--bid-file", str(tmp_path / "bid.parquet"),
          "--ask-file", str(tmp_path / "ask.parquet"), "--symbol", "US100", "--source", "test feed"])
    out = capsys.readouterr().out
    assert '"bars_per_session_mode": 7' in out and '"holiday_dates_dropped": 1' in out
    assert '"n_suspected_scale_breaks": 0' in out and '"unmatched_rows": 0' in out
    with MarketDataStore(db, read_only=True) as store:
        bars = store.load_hourly("US100")
        assert store.sources("US100") == ["test feed (resampled from 1-minute bars)"]
    hb = prepare_hourly_bars(bars)
    per_day = hb.groupby("session").size()
    assert len(per_day) == 10 and per_day.loc["2024-11-29"] == 4 and (per_day.drop(pd.Timestamp("2024-11-29")) == 7).all()
    first = hb.groupby("session").head(1)
    assert set(first.index.tz_convert(NY).strftime("%H:%M")) == {"09:30"}
    assert np.allclose(bars["spread_rel"], 1.8 / bars["close"], rtol=2e-3)
    with pytest.raises(ValueError, match="either --file, or both"):
        main(["ingest", "intraday", "--db", db, "--bid-file", str(tmp_path / "bid.parquet"), "--symbol", "X"])


def test_panama_chain_of_fine_bars_is_resampled(tmp_path, capsys):
    ts = pd.date_range("2024-02-26", "2024-03-23", freq="30min", tz="UTC", inclusive="left")
    spot = 17_000 + np.arange(len(ts)) * 0.5
    rows = []
    for name, expiry, basis in (("NQH4", "2024-03-15", 40.0), ("NQM4", "2024-06-21", 110.0)):
        for t, s in zip(ts, spot):
            if t.tz_localize(None) <= pd.Timestamp(expiry) + pd.Timedelta(days=1):
                rows.append({"contract": name, "expiry": expiry, "ts": t, "open": s + basis, "high": s + basis + 1,
                             "low": s + basis - 1, "close": s + basis + 0.5, "volume": 10.0})
    db = str(tmp_path / "f.duckdb")
    with MarketDataStore(db) as store:
        store.upsert_futures("NQ", pd.DataFrame(rows), "test chain")
    main(["ingest", "panama", "--db", db, "--root", "NQ", "--symbol", "NQ_C", "--resample"])
    assert '"bars_per_session_mode": 7' in capsys.readouterr().out
    with MarketDataStore(db, read_only=True) as store:
        bars = store.load_hourly("NQ_C")
    hb = prepare_hourly_bars(bars)
    assert hb.groupby("session").size().eq(7).all() and "raw_close" in bars
    assert np.abs(np.diff(bars["close"])).max() < 100                 # the roll gap of 70 points is adjusted away
    assert (bars["close"] - bars["raw_close"]).abs().max() == pytest.approx(70.0)
