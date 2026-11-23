"""Tier-1 scheduled US events, 2015-2025.

FOMC   scheduled meetings only (unscheduled meetings, notation votes and the
       cancelled March 2020 meeting are not events known in advance); the
       statement is released at 14:00 ET on the last day of the meeting.
       2015-2020 from the Federal Reserve's historical pages, 2017-2025 from
       its calendar feed; the two are cross-checked where they overlap.
NFP    Employment Situation release dates from ALFRED, 08:30 ET.
CPI    Consumer Price Index release dates from ALFRED, 08:30 ET.
"""
import json
import re
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], start=1)}

hist = set()
for year in range(2015, 2021):
    html = open(f"{src}/fomchistorical{year}.htm", encoding="utf-8", errors="replace").read()
    for h in re.findall(r"<h5[^>]*>([^<]*)</h5>", html):
        m = re.match(r"\s*([A-Z][a-z]+)(?:/([A-Z][a-z]+))?\s+(\d+)(?:-(\d+))?\s+Meeting\s+-\s+(\d{4})", h)
        if not m:
            continue                                   # unscheduled, cancelled, notation votes, conference calls
        mon1, mon2, d1, d2, yr = m.groups()
        last_month = {k[:3]: v for k, v in MONTHS.items()}[(mon2 or mon1)[:3]]
        hist.add(pd.Timestamp(int(yr), last_month, int(d2 or d1)))

j = json.loads(open(f"{src}/fed_calendar.json", "rb").read().decode("utf-8-sig"))
feed = set()
for e in j["events"]:
    if e.get("type") == "FOMC" and e.get("title", "").strip().lower() == "fomc meeting":
        y, mth = e["month"].split("-")
        day = int(re.findall(r"\d+", e["days"])[-1])   # "16-17" -> 17
        feed.add(pd.Timestamp(int(y), int(mth), day))

# 2021 onward: the Federal Reserve's meeting calendar page
M3 = {k[:3]: v for k, v in MONTHS.items()}
page = set()
html = open(f"{src}/fomccalendars.htm", encoding="utf-8", errors="replace").read()
parts = re.split(r'<h4><a id="\d+">(20\d\d) FOMC Meetings', html)
for year, body in zip(parts[1::2], parts[2::2]):
    for mon, days in re.findall(r'fomc-meeting__month[^>]*>\s*<strong>([^<]*)</strong>.*?fomc-meeting__date[^>]*>([^<]*)<',
                                body, flags=re.S):
        if "(" in days:                                # notation vote / unscheduled
            continue
        last_month = M3[mon.split("/")[-1].strip()[:3]]
        page.add(pd.Timestamp(int(year), last_month, int(re.findall(r"\d+", days)[-1])))

for name, x, y, years in (("pages vs feed", hist, feed, range(2017, 2021)), ("calendar vs feed", page, feed, (2021, 2022, 2025))):
    a_ = {d for d in x if d.year in years}
    b_ = {d for d in y if d.year in years}
    print(f"FOMC cross-check {name} {list(years)[0]}-{list(years)[-1]}: {len(a_)} vs {len(b_)}, "
          f"differences {sorted(str(t.date()) for t in a_ ^ b_)}")
fomc = sorted({d for d in hist if d.year <= 2020} | {d for d in page if 2021 <= d.year <= 2025})
print("FOMC per year:", pd.Series(1, index=pd.DatetimeIndex(fomc)).groupby(lambda d: d.year).sum().to_dict())


def alfred_dates(path: str) -> list[pd.Timestamp]:
    lines = open(path).read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "Release Dates:")
    return sorted({pd.Timestamp(ln.strip()) for ln in lines[start + 1:] if re.fullmatch(r"\d{4}-\d{2}-\d{2}", ln.strip())})


nfp, cpi = alfred_dates(f"{src}/alfred_rid50.txt"), alfred_dates(f"{src}/alfred_rid10.txt")
rows = [(d + pd.Timedelta(hours=14), "FOMC", 1) for d in fomc]
rows += [(d + pd.Timedelta(hours=8, minutes=30), "NFP", 1) for d in nfp]
rows += [(d + pd.Timedelta(hours=8, minutes=30), "CPI", 1) for d in cpi]
ev = pd.DataFrame(rows, columns=["ts", "name", "tier"])
ev = ev[(ev.ts >= "2015-01-01") & (ev.ts < "2025-08-01")].sort_values("ts")
print(ev.groupby([ev.ts.dt.year, "name"]).size().unstack().to_string())
ev.to_csv(out + "/tier1_events.csv", index=False)
print("events written", len(ev))
