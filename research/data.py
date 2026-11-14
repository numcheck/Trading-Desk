"""
Data layer - free feeds (Yahoo chart API, no key).

REAL LIMITS MEASURED on 2026-11-13:
    1m  : 60 days of history max, 7 d/request
    5m  : 60 days of history max        <-- unusable for 3 years
    15m : 60 days of history max
    30m : 60 days of history max
    1h  : 730 days of history max       <-- the only long intraday granularity
    1d  : unlimited

Hence the strategy retained:
  * ES=F / NQ=F DAILY over 6 years     -> amplitude and location study
  * SPY / QQQ at 1h over 730 days      -> timing, real time, backtest
  * ES=F / SPY at 5m over 60 days      -> high-resolution control

SPY 1h bars with includePrePost=false align exactly on the session
(09:30, 10:30, ..., 15:30) plus a spurious 16:00 bar capturing post-close
trading: it is REMOVED. The reconstruction is validated against the exchange's
daily OHLC (see validate()).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("research/cache")
CACHE.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
ET = "America/New_York"
LIMITS = {"1m": 25, "5m": 60, "15m": 60, "30m": 60, "1h": 730, "1d": 10000}


def _get(url: str, tries: int = 5) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9"})
    last = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 422) and k < tries - 1:
                time.sleep(3.0 * (k + 1))
                continue
            raise RuntimeError(f"Yahoo HTTP {e.code}: {e.read()[:200]!r}") from e
        except Exception as e:                      # noqa: BLE001
            last = e
            time.sleep(2.0 * (k + 1))
    raise RuntimeError(f"Yahoo: failed after {tries} attempts ({last})")


def yahoo(symbol: str, interval: str, days: int, prepost: bool = False) -> pd.DataFrame:
    """Downloads a series, indexed in New York time."""
    days = min(days, LIMITS[interval])
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    out, cur = [], start
    chunk = {"1m": 7, "5m": 55, "15m": 55, "30m": 55, "1h": 700, "1d": 4000}[interval]
    while cur < end:
        nxt = min(cur + timedelta(days=chunk), end)
        q = urllib.parse.urlencode({
            "period1": int(cur.timestamp()), "period2": int(nxt.timestamp()),
            "interval": interval, "includePrePost": str(prepost).lower()})
        js = _get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{q}")
        res = (js.get("chart") or {}).get("result") or []
        if res and (res[0].get("timestamp") or []):
            r = res[0]
            q0 = (r.get("indicators") or {}).get("quote", [{}])[0]
            out.append(pd.DataFrame({
                "open": q0.get("open"), "high": q0.get("high"), "low": q0.get("low"),
                "close": q0.get("close"), "volume": q0.get("volume")},
                index=pd.to_datetime(r["timestamp"], unit="s", utc=True)))
        cur = nxt
        time.sleep(0.4)
    df = pd.concat(out)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df.high >= df.low) & (df.close > 0) & (df.high >= df[["open", "close"]].max(1))
            & (df.low <= df[["open", "close"]].min(1))]
    return df.tz_convert(ET)


def daily(symbol: str, years: int = 6) -> pd.DataFrame:
    """Daily OHLC (ground truth for session H/L)."""
    f = CACHE / f"{symbol.replace('=','_')}_1d_{years}y.pkl"
    if f.exists():
        return pd.read_pickle(f)
    d = yahoo(symbol, "1d", int(years * 365.25))
    d["date"] = d.index.date
    d = d.groupby("date").agg(open=("open", "first"), high=("high", "max"),
                              low=("low", "min"), close=("close", "last"),
                              volume=("volume", "sum"))
    d.index = pd.to_datetime(d.index)
    d = d[d.volume.fillna(0) > 0]
    d = d[d.high > d.low]
    d.to_pickle(f)
    return d


def bars_1h(symbol: str, days: int = 730) -> pd.DataFrame:
    """Reconstructed RTH 1h bars (7 bars: 09:30 -> 16:00)."""
    f = CACHE / f"{symbol.replace('=','_')}_1h.pkl"
    if f.exists():
        return pd.read_pickle(f)
    d = yahoo(symbol, "1h", days, prepost=False)
    mins = d.index.hour * 60 + d.index.minute
    d = d[(mins >= 9 * 60 + 30) & (mins < 16 * 60)].copy()   # remove the 16:00 bar
    mins = d.index.hour * 60 + d.index.minute                # recompute after filtering
    d["date"] = pd.to_datetime(d.index.date)                 # Timestamp, not datetime.date
    d["slot"] = (mins - (9 * 60 + 30)) // 60                  # 0..6
    d = d[d.slot <= 6]
    d.to_pickle(f)
    return d


def sessions_1h(symbol: str, days: int = 730, min_slots: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates into sessions. Returns (RTH 1h bars, per-session table)."""
    b = bars_1h(symbol, days)
    g = b.groupby("date")
    sess = pd.DataFrame({
        "open": g.open.first(), "high": g.high.max(), "low": g.low.min(),
        "close": g.close.last(), "volume": g.volume.sum(), "n_slots": g.size(),
        "i_high": g.apply(lambda x: int(np.argmax(x.high.values))),
        "i_low": g.apply(lambda x: int(np.argmin(x.low.values))),
    })
    sess.index = pd.to_datetime(sess.index)
    sess = sess[sess.n_slots >= min_slots]
    sess["range"] = sess.high - sess.low

    def _rv(x):
        c = np.r_[x.open.iloc[0], x.close.values]
        return float(np.sum(np.diff(np.log(c)) ** 2))
    sess["rv"] = b.groupby("date").apply(_rv).reindex(sess.index)
    sess["t_high"] = (sess.i_high + 0.5) / 7.0     # middle of the bar, normalised
    sess["t_low"] = (sess.i_low + 0.5) / 7.0
    sess["dow"] = sess.index.dayofweek
    return b, sess


def validate(symbol: str, days: int = 730) -> dict:
    """The 1h reconstruction must recover the exchange's daily OHLC."""
    try:
        d = daily(symbol, 3)
    except Exception as e:                            # noqa: BLE001
        return {"symbol": symbol, "error": str(e)}
    _, s = sessions_1h(symbol, days)
    j = s.join(d[["open", "high", "low", "close"]], how="inner", rsuffix="_d")
    if len(j) < 20:
        return {"symbol": symbol, "n": int(len(j))}
    rel = lambda a, b: float(np.median(np.abs(a - b) / np.maximum(b, 1e-9)))
    out = {"symbol": symbol, "n": int(len(j)),
           "med_rel_open": rel(j.open, j.open_d), "med_rel_high": rel(j.high, j.high_d),
           "med_rel_low": rel(j.low, j.low_d), "med_rel_close": rel(j.close, j.close_d),
           "match_high_2bp": float(np.mean(np.abs(j.high - j.high_d) / j.high_d < 2e-4)),
           "match_low_2bp": float(np.mean(np.abs(j.low - j.low_d) / j.low_d < 2e-4)),
           "range_ratio": float(np.median(j.range / (j.high_d - j.low_d)))}
    return out


if __name__ == "__main__":
    for sym in ("SPY", "QQQ", "IWM"):
        for k, v in validate(sym).items():
            print(f"{k:16s} {v}")
        print("-" * 50)
