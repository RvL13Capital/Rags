"""NYSE regular-session calendar for anchoring intraday bars.

Sources that quote around the clock (index CFDs, futures) print bars on US
market holidays and after an early close.  Those are not cash sessions and
must not become sessions in the store.

The calendar comes from the maintained ``exchange_calendars`` package (XNYS)
when it is installed (``pip install 'ndxswing[calendar]'``).  Without it a
rule-based fallback is used; ``tests/test_ingest_intraday.py`` checks the
fallback against XNYS for 2012-2026 whenever the package is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr, USMemorialDay, USPresidentsDay,
    USThanksgivingDay, nearest_workday, sunday_to_monday,
)

REGULAR_CLOSE_MIN = 16 * 60
EARLY_CLOSE_MIN = 13 * 60
#: unscheduled full-day closures: Hurricane Sandy, national days of mourning
SPECIAL_CLOSURES = ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09")


@dataclass(frozen=True)
class SessionCalendar:
    sessions: pd.DatetimeIndex        # naive New York dates with a regular session
    close_minute: pd.Series           # minutes after midnight ET at which each session closes
    source: str

    def close_of(self, dates: pd.DatetimeIndex) -> pd.Series:
        return self.close_minute.reindex(dates)


class _NyseRules(AbstractHolidayCalendar):
    rules = [
        # a Saturday New Year's Day is not observed on the preceding Friday
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def rule_based_calendar(start: str, end: str) -> SessionCalendar:
    """Fallback: scheduled holidays by rule, the known special closures and
    the three recurring 13:00 early closes."""
    days = pd.bdate_range(start, end)
    closed = _NyseRules().holidays(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end) + pd.Timedelta(days=7))
    closed = closed.union(pd.DatetimeIndex(SPECIAL_CLOSURES))
    sessions = days.difference(closed)
    close = pd.Series(REGULAR_CLOSE_MIN, index=sessions)
    thanksgiving = USThanksgivingDay.dates(pd.Timestamp(start) - pd.Timedelta(days=7), pd.Timestamp(end))
    early = [d + pd.Timedelta(days=1) for d in thanksgiving]
    for year in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        eve, jul3 = pd.Timestamp(year, 12, 24), pd.Timestamp(year, 7, 3)
        early.append(eve)                                   # Christmas Eve, when it is a session
        if jul3.dayofweek <= 3:                             # 3 July, Monday to Thursday
            early.append(jul3)
    close.loc[close.index.intersection(pd.DatetimeIndex(early))] = EARLY_CLOSE_MIN
    return SessionCalendar(sessions, close, "built-in NYSE rules (install exchange_calendars for the maintained XNYS calendar)")


def nyse_calendar(start: str, end: str, prefer_package: bool = True) -> SessionCalendar:
    start, end = str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())
    if prefer_package:
        try:
            import exchange_calendars as xc
        except ImportError:
            xc = None
        if xc is not None:
            pad_lo = str((pd.Timestamp(start) - pd.Timedelta(days=10)).date())
            pad_hi = str((pd.Timestamp(end) + pd.Timedelta(days=10)).date())
            cal = xc.get_calendar("XNYS", start=pad_lo, end=pad_hi)
            sessions = pd.DatetimeIndex(cal.sessions_in_range(start, end)).tz_localize(None)
            closes = pd.DatetimeIndex(cal.closes.loc[sessions]).tz_convert("America/New_York")
            close = pd.Series(closes.hour * 60 + closes.minute, index=sessions)
            return SessionCalendar(sessions, close, f"exchange_calendars XNYS {xc.__version__}")
    return rule_based_calendar(start, end)
