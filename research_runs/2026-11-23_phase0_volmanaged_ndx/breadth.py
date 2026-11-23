"""How many independent volatility regimes does a basket of equity indices hold?

Eigenvalue participation ratio of the correlation matrix of weekly log
changes in public equity volatility indices (FRED).  No strategy is run and
no index price is loaded.

Usage: breadth.py <out dir>
"""
import json
import sys

import numpy as np
import pandas as pd

from ndxswing.analytics.relative import participation_ratio
from ndxswing.data.ingest import fetch_fred_csv

out = sys.argv[1]
SERIES = {"VXNCLS": "NASDAQ-100", "VIXCLS": "S&P 500", "VXDCLS": "Dow Jones", "RVXCLS": "Russell 2000",
          "VXEEMCLS": "Emerging markets", "VXEWZCLS": "Brazil", "VXFXICLS": "China"}
raw = pd.DataFrame({name: fetch_fred_csv(sid, "2015-01-01") for sid, name in SERIES.items()})
raw = raw[raw.index <= "2025-07-24"]
res = {}
for label, cols, freq in (
        ("4 US indices, weekly", ["NASDAQ-100", "S&P 500", "Dow Jones", "Russell 2000"], "W-FRI"),
        ("6 indices (US + emerging markets + Brazil), weekly", ["NASDAQ-100", "S&P 500", "Dow Jones", "Russell 2000", "Emerging markets", "Brazil"], "W-FRI"),
        ("7 indices incl. China (to February 2022), weekly", list(SERIES.values()), "W-FRI"),
        ("6 indices, monthly", ["NASDAQ-100", "S&P 500", "Dow Jones", "Russell 2000", "Emerging markets", "Brazil"], "ME"),
        ("6 indices, weekly, levels of log volatility", ["NASDAQ-100", "S&P 500", "Dow Jones", "Russell 2000", "Emerging markets", "Brazil"], "level")):
    x = np.log(raw[cols].dropna())
    chg = x.resample("W-FRI").last().dropna() if freq == "level" else x.resample(freq).last().diff().dropna()
    c = chg.corr()
    lam = np.sort(np.linalg.eigvalsh(c.to_numpy()))[::-1]
    off = c.to_numpy()[~np.eye(len(cols), dtype=bool)]
    res[label] = {"n_indices": len(cols), "observations": int(len(chg)), "from": str(chg.index[0].date()), "to": str(chg.index[-1].date()),
                  "participation_ratio": round(participation_ratio(c.to_numpy()), 2), "mean_correlation": round(float(off.mean()), 3),
                  "min_correlation": round(float(off.min()), 3), "first_eigenvalue_share": round(float(lam[0] / lam.sum()), 3)}
    if label.startswith("6 indices (US"):
        c.round(3).to_csv(f"{out}/vol_index_correlations_weekly.csv")
        print(c.round(2).to_string())
    print(label, res[label])
json.dump(res, open(f"{out}/breadth.json", "w"), indent=1)
