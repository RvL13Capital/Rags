"""Volatility-managed index exposure for a small EUR retail account.

Three layers, kept apart so each can be checked on its own:

1. measurement  - realised variance, bipower variation, jump variation and
                  downside semivariance of one session from 1-minute bars;
                  causal variance forecasts built from them;
2. weights      - target weight c / forecast, with the scalar c either fitted
                  on the whole sample (Moreira & Muir 2017, look-ahead) or
                  re-estimated on an expanding window (Cederburg et al. 2020);
3. accounting   - what a EUR account at a CFD broker, or a buy-and-hold UCITS
                  ETF, is actually worth after spread, conversion mark-up,
                  overnight funding, dividends and German tax.

Timing convention of the account simulators: a target decided on data through
the close of day t is traded at the open of day t+1.

Tax model (simplified where stated in the functions): CFD income is taxed at
the flat rate per calendar year on net realised amounts with losses carried
forward; an equity ETF is taxed on 70% of its gain when sold.  "Liquidation
basis" values deduct the costs and tax that closing the account that day would
trigger, so the last value is the post-tax terminal wealth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..analytics.relative import long_run_covariance, newey_west_lags

# --------------------------------------------------------------------------- #
# 1. Measurement
# --------------------------------------------------------------------------- #


def realised_measures(open_1m: np.ndarray, close_1m: np.ndarray, step: int = 5) -> dict[str, float]:
    """Realised measures of one session from its 1-minute bars.

    sum_r1   sum of 1-minute log returns from the session open to its close
    rv1      realised variance on 1-minute returns
    rv5      realised variance on ``step``-minute returns
    bv5      bipower variation (pi / 2) n / (n - 1) sum |r_i| |r_{i-1}|: robust to jumps
    jv5      jump variation max(rv5 - bv5, 0)
    rsm5     downside realised semivariance, sum of squared negative returns
    rsp5     upside realised semivariance (rsm5 + rsp5 = rv5)
    """
    o, c = np.asarray(open_1m, dtype=float), np.asarray(close_1m, dtype=float)
    if len(c) == 0:
        raise ValueError("empty session")
    p = np.log(np.r_[o[0], c])
    r1 = np.diff(p)
    grid = np.arange(0, len(p), step)
    if grid[-1] != len(p) - 1:
        grid = np.r_[grid, len(p) - 1]
    r5 = np.diff(p[grid])
    n = len(r5)
    rv5 = float(np.sum(r5**2))
    bv5 = float(np.pi / 2.0 * n / (n - 1) * np.sum(np.abs(r5[1:]) * np.abs(r5[:-1]))) if n > 1 else rv5
    return {"sum_r1": float(r1.sum()), "rv1": float(np.sum(r1**2)), "rv5": rv5, "bv5": bv5,
            "jv5": max(rv5 - bv5, 0.0), "rsm5": float(np.sum(r5[r5 < 0] ** 2)), "rsp5": float(np.sum(r5[r5 > 0] ** 2))}


def variance_signals(daily: pd.DataFrame, window: int = 20, har_min_obs: int = 252) -> pd.DataFrame:
    """Causal forecasts of the next session's total (close-to-close) variance.
    Row t uses data through the close of session t only.

    rv20      mean squared close-to-close log return over ``window`` sessions (the v8 RV20, as a daily variance)
    rvi20     mean of intraday realised variance plus squared overnight gap
    bv20      mean of bipower variation plus squared overnight gap (jumps removed from the intraday part)
    rsm20     twice the mean downside part (downside semivariance plus squared negative gaps)
    har       HAR regression of next-day total variance on its daily, weekly and monthly means,
              coefficients re-estimated every day on all data up to that day
    """
    d = daily
    gap2 = d["r_gap"].fillna(0.0) ** 2
    tv = d["rv5"] + gap2
    out = pd.DataFrame(index=d.index)
    out["rv20"] = (d["r_cc"] ** 2).rolling(window, min_periods=window).mean()
    out["rvi20"] = tv.rolling(window, min_periods=window).mean()
    out["bv20"] = (d["bv5"] + gap2).rolling(window, min_periods=window).mean()
    down = d["rsm5"] + gap2.where(d["r_gap"] < 0, 0.0)
    out["rsm20"] = 2.0 * down.rolling(window, min_periods=window).mean()

    # HAR with expanding-window coefficients: the regression fitted at t uses targets up to tv_t only
    x = pd.DataFrame({"c": 1.0, "d": tv, "w": tv.rolling(5).mean(), "m": tv.rolling(22).mean()})
    X = x.to_numpy()
    y_next = tv.shift(-1).to_numpy()
    har = np.full(len(d), np.nan)
    xtx, xty, n_obs = np.zeros((4, 4)), np.zeros(4), 0
    floor = tv.expanding(min_periods=window).mean().to_numpy() * 0.10      # a variance forecast cannot be ~zero or negative
    for t in range(len(d)):
        if t >= 1 and np.isfinite(X[t - 1]).all() and np.isfinite(y_next[t - 1]):
            xtx += np.outer(X[t - 1], X[t - 1])                              # pair (x_{t-1}, tv_t) is known at t
            xty += X[t - 1] * y_next[t - 1]
            n_obs += 1
        if n_obs >= har_min_obs and np.isfinite(X[t]).all():
            beta = np.linalg.solve(xtx, xty)
            har[t] = max(float(X[t] @ beta), floor[t])
    out["har"] = har
    return out


# --------------------------------------------------------------------------- #
# 2. Weights
# --------------------------------------------------------------------------- #


def managed_weights(forecast: pd.Series, returns: pd.Series, scalar: str = "causal", power: float = 1.0,
                    min_obs: int = 252, sample: slice | None = None) -> pd.DataFrame:
    """Target weight for the NEXT session, known at the close of each row.

    raw_t = forecast_t ** -power   (power 1: inverse variance, 0.5: inverse volatility)
    w_t   = c_t * raw_t

    c makes the unlevered managed series m_s = raw_{s-1} r_s as volatile as r:
    ``scalar="insample"`` uses one c from the whole ``sample`` (a look-ahead:
    it cannot be known in real time); ``scalar="causal"`` uses, at each t, the
    returns up to and including t (at least ``min_obs`` of them)."""
    raw = forecast.astype(float) ** (-power)
    m = raw.shift(1) * returns
    df = pd.DataFrame({"raw": raw, "r": returns, "m": m})
    if scalar == "insample":
        s = df if sample is None else df.loc[sample]
        ok = s["m"].notna() & s["r"].notna()
        c = pd.Series(float(s.loc[ok, "r"].std(ddof=1) / s.loc[ok, "m"].std(ddof=1)), index=df.index)
    elif scalar == "causal":
        ok = df["m"].notna() & df["r"].notna()
        sd_r = df["r"].where(ok).expanding(min_periods=min_obs).std(ddof=1)
        sd_m = df["m"].where(ok).expanding(min_periods=min_obs).std(ddof=1)
        c = sd_r / sd_m
    else:
        raise ValueError("scalar must be 'insample' or 'causal'")
    df["c"] = c
    df["w"] = c * raw
    return df[["raw", "c", "w"]]


# --------------------------------------------------------------------------- #
# 3. Accounting
# --------------------------------------------------------------------------- #

#: Basiszins published by the German finance ministry for the Vorabpauschale (negative years give none)
BASISZINS = {2018: 0.0087, 2019: 0.0052, 2020: 0.0007, 2021: 0.0, 2022: 0.0, 2023: 0.0255, 2024: 0.0229, 2025: 0.0253}


@dataclass(frozen=True)
class RetailCosts:
    spread_points: float = 1.8          # quoted spread of the index CFD, index points
    fx_fee: float = 0.007               # mark-up on converting realised P&L, dividends and funding into EUR
    funding_markup: float = 0.04        # overnight funding = (benchmark + markup) / day_count on the notional
    day_count: int = 360
    dividend_credit: float = 1.0        # share of the index dividend credited to a long CFD
    lot_step: float = 0.001


@dataclass(frozen=True)
class GermanTax:
    """Abgeltungsteuer incl. solidarity surcharge.  ``allowance_*`` is the
    saver's allowance still unused by other income (0 = already used up)."""

    enabled: bool = True
    rate: float = 0.26375
    equity_exemption: float = 0.30      # Teilfreistellung of an equity fund: 18.4625% effective
    allowance_to_2022: float = 0.0
    allowance_from_2023: float = 0.0
    vorabpauschale: bool = False

    def allowance(self, year: int) -> float:
        return self.allowance_from_2023 if year >= 2023 else self.allowance_to_2022

    def due(self, taxable: float, year: int) -> float:
        return self.rate * max(taxable - self.allowance(year), 0.0) if self.enabled else 0.0


@dataclass(frozen=True)
class CashSweep:
    """Idle EUR swept into an overnight-rate ETF: index = overnight rate + ``index_spread``, less ``ter``;
    ``ticket`` per transfer; held only while the net rate is positive."""

    index_spread: float = 0.00085
    ter: float = 0.0010
    ticket: float = 1.0
    half_spread: float = 0.0001

    def net_rate(self, overnight: float) -> float:
        return max(overnight + self.index_spread - self.ter, 0.0)


@dataclass
class MarketDays:
    """Per-session inputs of the account simulators."""

    index: pd.DatetimeIndex
    open: np.ndarray            # index level at the session open (quote currency: USD)
    close: np.ndarray
    fx: np.ndarray              # USD per EUR
    usd_rate: np.ndarray        # annual, decimal: funding benchmark
    eur_rate: np.ndarray        # annual, decimal: overnight EUR rate
    div: np.ndarray             # gross dividend return of the index going ex at this session's open
    nights: np.ndarray = field(default=None)   # calendar days since the previous session

    def __post_init__(self) -> None:
        if self.nights is None:
            d = np.diff(self.index.values).astype("timedelta64[D]").astype(float)
            self.nights = np.r_[1.0, d]

    @classmethod
    def from_frames(cls, daily: pd.DataFrame, macro: pd.DataFrame, start: str | pd.Timestamp | None = None,
                    end: str | pd.Timestamp | None = None) -> "MarketDays":
        d = daily.join(macro, how="inner")
        if start is not None:
            d = d[d.index >= pd.Timestamp(start)]
        if end is not None:
            d = d[d.index <= pd.Timestamp(end)]
        return cls(d.index, d["open"].to_numpy(float), d["close"].to_numpy(float), d["eurusd"].to_numpy(float),
                   d["usd_rate"].to_numpy(float), d["estr"].to_numpy(float), d["div_yield_day"].to_numpy(float))

    def __len__(self) -> int:
        return len(self.index)


@dataclass
class AccountResult:
    frame: pd.DataFrame          # per session: equity (mark-to-mid), liq_pre, liq_post, weight, units
    initial: float
    totals: dict                 # cumulative spread, conversion fees, funding, dividends, interest, tax, rebalances

    def returns(self, column: str = "liq_post") -> pd.Series:
        v = self.frame[column].to_numpy(float)
        return pd.Series(v / np.r_[self.initial, v[:-1]] - 1.0, index=self.frame.index, name=column)

    @property
    def terminal(self) -> float:
        return float(self.frame["liq_post"].iloc[-1])


def _year_end(index: pd.DatetimeIndex) -> np.ndarray:
    y = index.year.to_numpy()
    return np.r_[y[1:] != y[:-1], True]


def simulate_cfd_account(mkt: MarketDays, target: np.ndarray, *, initial: float = 5000.0, deadband: float = 0.10,
                         w_max: float = 1.0, costs: RetailCosts = RetailCosts(), tax: GermanTax = GermanTax(),
                         funding: str = "none", sweep: CashSweep | None = None, rebalance: bool = True) -> AccountResult:
    """EUR account holding a long index CFD at a managed weight.

    ``target[t]`` is the weight to hold from the open of session t (NaN: no
    view).  The first finite target opens the position; afterwards a trade
    happens only if ``rebalance`` and the target differs from the current
    weight by more than ``deadband``.  ``rebalance=False`` is buy-and-hold.

    funding  "none"   1:1 mode: nothing is charged, and the cap must not exceed 1
             "cliff"  nights on which notional / equity exceeded 1 at the close are charged
                      (benchmark + markup) / day_count on the WHOLE notional
    Idle cash earns nothing unless ``sweep`` is given, in which case the cash
    not needed to back the position is moved to an overnight-rate ETF on
    rebalancing days.

    P&L accrues in USD on the units held and is converted at the session's
    EURUSD; conversion of realised amounts pays ``fx_fee``.  Realised gains use
    the average entry price.
    """
    if funding not in ("none", "cliff"):
        raise ValueError("funding must be 'none' or 'cliff'")
    if funding == "none" and w_max > 1.0 + 1e-12:
        raise ValueError("a weight above 1 needs the leveraged mode (funding='cliff')")
    n = len(mkt)
    hs = 0.5 * costs.spread_points
    yend = _year_end(mkt.index)
    years = mkt.index.year.to_numpy()
    cash, swept, units, pavg = float(initial), 0.0, 0.0, 0.0
    ytd, carry, lev_prev, opened = 0.0, 0.0, 0.0, False
    tot = {"spread": 0.0, "fx_fee": 0.0, "funding": 0.0, "dividends": 0.0, "interest": 0.0, "tax": 0.0,
           "rebalances": 0, "turnover": 0.0, "nights_levered": 0}
    rec = np.zeros((n, 5))

    def realise(amount_usd: float, x: float) -> float:
        """Convert a realised USD amount into EUR cash, pay the mark-up, book it for tax."""
        nonlocal cash, ytd
        eur = amount_usd / x
        fee = costs.fx_fee * abs(eur)
        cash += eur - fee
        ytd += eur - fee
        tot["fx_fee"] += fee
        return eur - fee

    for t in range(n):
        o, c, x = mkt.open[t], mkt.close[t], mkt.fx[t]
        if t > 0 and units > 0:
            div_usd = units * mkt.div[t] * mkt.close[t - 1] * costs.dividend_credit
            if div_usd != 0.0:
                tot["dividends"] += realise(div_usd, x)
            if funding == "cliff" and lev_prev > 1.0 + 1e-9:
                fin_usd = units * mkt.close[t - 1] * (mkt.usd_rate[t - 1] + costs.funding_markup) / costs.day_count * mkt.nights[t]
                tot["funding"] -= realise(-fin_usd, x)
                tot["nights_levered"] += int(mkt.nights[t])
        if sweep is not None and swept > 0 and t > 0:
            intr = swept * sweep.net_rate(mkt.eur_rate[t - 1]) * mkt.nights[t] / 360.0
            swept += intr
            ytd += intr
            tot["interest"] += intr

        equity = cash + swept + units * (o - pavg) / x
        w_cur = units * o / x / equity if equity > 0 else 0.0
        tgt = target[t]
        if np.isfinite(tgt) and (not opened or (rebalance and abs(min(max(tgt, 0.0), w_max) - w_cur) > deadband)):
            tgt = min(max(float(tgt), 0.0), w_max)
            new_units = np.floor(tgt * equity * x / o / costs.lot_step + 1e-9) * costs.lot_step
            du = new_units - units
            if du > 0:
                pavg = (units * pavg + du * (o + hs)) / new_units
            elif du < 0:
                realise(-du * ((o - hs) - pavg), x)
            if du != 0:
                tot["spread"] += abs(du) * hs / x
                tot["turnover"] += abs(du) * o / x
                tot["rebalances"] += 1
                units = new_units
                if units <= 1e-12:
                    units, pavg = 0.0, 0.0
            opened = True
            if sweep is not None:
                # cash beyond what backs the position goes to (or comes back from) the overnight-rate ETF
                eq_now = cash + swept + units * (o - pavg) / x
                want = max(eq_now - units * o / x, 0.0) if sweep.net_rate(mkt.eur_rate[t]) > 0 else 0.0
                move = want - swept
                if abs(move) > 50.0:                              # not worth a ticket below this
                    cost = sweep.ticket + sweep.half_spread * abs(move)
                    cash -= move + cost
                    swept += move
                    ytd -= cost
                    tot["spread"] += cost

        unreal = units * (c - pavg) / x
        equity = cash + swept + unreal
        lev_prev = units * c / x / equity if equity > 0 else 0.0
        # what closing the account today would leave
        close_usd = units * ((c - hs) - pavg)
        close_eur = close_usd / x
        close_fee = costs.fx_fee * abs(close_eur) + (sweep.ticket if sweep is not None and swept > 0 else 0.0)
        liq_pre = cash + swept + close_eur - close_fee
        liq_post = liq_pre - tax.due(ytd + carry + close_eur - close_fee, int(years[t]))
        if yend[t]:
            if t == n - 1 and units > 0:                           # actual liquidation at the end of the sample
                realise(close_usd, x)
                tot["spread"] += units * hs / x
                units, pavg = 0.0, 0.0
                if sweep is not None and swept > 0:
                    cash += swept - sweep.ticket
                    ytd -= sweep.ticket
                    swept = 0.0
            base = ytd + carry
            due = tax.due(base, int(years[t]))
            cash -= due
            tot["tax"] += due
            carry = min(base, 0.0) if tax.enabled else 0.0
            ytd = 0.0
        rec[t] = (equity, liq_pre, liq_post, lev_prev, units)
    frame = pd.DataFrame(rec, index=mkt.index, columns=["equity", "liq_pre", "liq_post", "weight", "units"])
    return AccountResult(frame, float(initial), tot)


def simulate_etf_buy_and_hold(mkt: MarketDays, *, initial: float = 5000.0, ter: float = 0.0030,
                              dividend_withholding: float = 0.15, half_spread: float = 0.0003, ticket: float = 1.0,
                              tax: GermanTax = GermanTax()) -> AccountResult:
    """Accumulating equity UCITS ETF on the index, bought at the first open and held.

    The fund earns the index price return plus dividends net of the
    withholding it suffers, less the TER; its EUR price carries EURUSD on the
    whole holding.  Tax: 70% of the gain at the flat rate when sold.  With
    ``tax.vorabpauschale`` the annual advance lump sum (70% of the Basiszins on
    the value at the start of the year, capped at the year's gain) is taxed
    each January by selling units and credited against the final gain.
    """
    n = len(mkt)
    fund = np.empty(n)
    level = mkt.close[0] / mkt.open[0]                              # bought at the first open
    fund[0] = level
    for t in range(1, n):
        level *= (mkt.close[t] / mkt.close[t - 1] + mkt.div[t] * (1.0 - dividend_withholding)) * (
            1.0 - ter * mkt.nights[t] / 365.0)
        fund[t] = level
    price = fund * mkt.open[0] / mkt.fx                             # EUR per unit at each close
    price_open0 = mkt.open[0] / mkt.fx[0]
    units = (initial - ticket) / (price_open0 * (1.0 + half_spread))
    basis, vp_credit = float(initial), 0.0                          # acquisition cost incl. ticket and spread
    years = mkt.index.year.to_numpy()
    tot = {"spread": (initial - ticket) * half_spread / (1.0 + half_spread) + ticket, "fx_fee": 0.0, "funding": 0.0,
           "dividends": 0.0, "interest": 0.0, "tax": 0.0, "rebalances": 1, "turnover": initial, "nights_levered": 0}
    rec = np.zeros((n, 5))
    year_start_price, first_year = price_open0, int(years[0])
    for t in range(n):
        if tax.enabled and tax.vorabpauschale and t > 0 and years[t] != years[t - 1]:
            y = int(years[t - 1])
            held = 1.0 if y != first_year else (13 - mkt.index[0].month) / 12.0
            vp = min(units * year_start_price * BASISZINS.get(y, 0.0) * 0.7 * held,
                     max(units * (price[t - 1] - year_start_price), 0.0))
            due = tax.due(vp * (1.0 - tax.equity_exemption), int(years[t]))
            if due > 0:
                sold = due / (price[t - 1] * (1.0 - half_spread))
                basis *= 1.0 - sold / units
                vp_credit = (vp_credit + vp) * (1.0 - sold / units)
                units -= sold
                tot["tax"] += due
            else:
                vp_credit += vp
            year_start_price = price[t - 1]
        elif t > 0 and years[t] != years[t - 1]:
            year_start_price = price[t - 1]
        value = units * price[t]
        proceeds = value * (1.0 - half_spread) - ticket
        gain = proceeds - basis - vp_credit
        due = tax.due(gain * (1.0 - tax.equity_exemption), int(years[t])) if gain > 0 else 0.0
        rec[t] = (value, proceeds, proceeds - due, 1.0, units)
    tot["tax"] += due
    tot["spread"] += value * half_spread + ticket
    frame = pd.DataFrame(rec, index=mkt.index, columns=["equity", "liq_pre", "liq_post", "weight", "units"])
    return AccountResult(frame, float(initial), tot)


def buy_and_hold_cfd_closed_form(mkt: MarketDays, initial: float = 5000.0, w0: float = 1.0,
                                 lot_step: float = 0.001) -> np.ndarray:
    """Mark-to-mid EUR equity of a frictionless, untaxed 1:1 long opened at the
    first open with weight ``w0`` and never touched, written out directly:
    equity_t = E0 + units (close_t - open_0) / fx_t + dividends received.
    Used to check the simulator against an independent calculation."""
    units = np.floor(w0 * initial * mkt.fx[0] / mkt.open[0] / lot_step + 1e-9) * lot_step
    div_eur = np.cumsum(np.r_[0.0, units * mkt.div[1:] * mkt.close[:-1] / mkt.fx[1:]])
    return initial + units * (mkt.close - mkt.open[0]) / mkt.fx + div_eur


# --------------------------------------------------------------------------- #
# Frictionless series (mechanism tests and the literature comparison)
# --------------------------------------------------------------------------- #


def frictionless_excess(daily: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Daily index return including gross dividends, in excess of the USD cash rate."""
    nights = pd.Series(np.r_[1.0, np.diff(daily.index.values).astype("timedelta64[D]").astype(float)], index=daily.index)
    rf = macro["usd_rate"].shift(1).fillna(macro["usd_rate"]) * nights / 360.0
    return (np.expm1(daily["r_cc"]) + macro["div_yield_day"] - rf).rename("excess")


def hac_mean_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """t-statistic of the mean of a serially dependent series (Newey-West)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    lr = long_run_covariance((x - x.mean())[:, None], newey_west_lags(len(x)) if lags is None else lags)[0, 0]
    return float(x.mean() / np.sqrt(lr / len(x))) if lr > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Simulated returns with a known premium structure (size of the tests)
# --------------------------------------------------------------------------- #


def simulate_gjr_returns(n_paths: int, n_days: int, premium: str = "variance", sharpe_ann: float = 0.8,
                         vol_ann: float = 0.21, alpha: float = 0.03, gamma: float = 0.12, beta: float = 0.88,
                         nu: float = 8.0, seed: int = 0, burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """GJR-GARCH(1,1) daily returns with Student-t shocks, r_t = mu_t + sigma_t z_t.

    premium "variance"    mu_t = lambda sigma_t^2: the mean-variance optimal weight mu / sigma^2 is constant,
                          so scaling exposure by forecast variance has no value
            "volatility"  mu_t = lambda sigma_t: the conditional Sharpe ratio is constant and the optimal
                          weight lambda / sigma_t moves, so volatility timing does have value
    lambda is set so that the unconditional mean matches ``sharpe_ann`` at ``vol_ann``.
    Returns (returns, conditional variance), each (n_paths, n_days)."""
    rng = np.random.default_rng(seed)
    uncond = vol_ann**2 / 252.0
    omega = uncond * (1.0 - alpha - 0.5 * gamma - beta)
    if omega <= 0:
        raise ValueError("non-stationary GJR parameters")
    mean_day = sharpe_ann * vol_ann / 252.0
    total = n_days + burn
    z = rng.standard_t(nu, size=(n_paths, total)) * np.sqrt((nu - 2.0) / nu)
    s2 = np.full(n_paths, uncond)
    r, v = np.empty((n_paths, total)), np.empty((n_paths, total))
    lam_var = mean_day / uncond
    # E[sigma] sits a little below sqrt(E[sigma^2]); the exact level does not matter because the true
    # Sharpe ratios of this process are measured on a long simulation wherever they are needed
    lam_vol = mean_day / np.sqrt(uncond) / 0.97
    for t in range(total):
        sig = np.sqrt(s2)
        mu = lam_var * s2 if premium == "variance" else lam_vol * sig
        eps = sig * z[:, t]
        r[:, t], v[:, t] = mu + eps, s2
        s2 = omega + (alpha + gamma * (eps < 0)) * eps**2 + beta * s2
    return r[:, burn:], v[:, burn:]


def managed_from_returns(r: np.ndarray, window: int = 20, min_obs: int = 252, w_max: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Frictionless managed and buy-and-hold series from one return path, with
    the causal scalar and the RV20 forecast (the test statistic's inputs in
    the size simulations).  Returns (managed, benchmark) on the common span."""
    s = pd.Series(r)
    f = (s**2).rolling(window, min_periods=window).mean()
    w = managed_weights(f, s, "causal", 1.0, min_obs)["w"]
    if w_max is not None:
        w = w.clip(upper=w_max)
    m = (w.shift(1) * s).to_numpy()
    ok = np.isfinite(m)
    return m[ok], r[ok]
