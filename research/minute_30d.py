"""The free 30-day 1-minute test on the real futures (Yahoo, no key, no cost).

Yahoo serves 1-minute bars for the last ~25 days on ES=F and NQ=F. That is far too
little for a conclusion, but enough to check that the pipeline behaves on the actual
instrument rather than on an ETF proxy: full 390-bar RTH sessions, real volume, and an
amplitude ratio in the right ballpark.

Run: python -m research.minute_30d
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.data import yahoo

THEO_RUP = float(np.sqrt(2 / np.pi))
THEO_RANGE = float(np.sqrt(8 / np.pi))


def sessions_1m(sym: str, days: int = 25, min_bars: int = 300) -> pd.DataFrame:
    df = yahoo(sym, "1m", days, prepost=False)
    mins = df.index.hour * 60 + df.index.minute
    rth = df[(mins >= 570) & (mins < 960)].copy()      # 09:30 -> 16:00 ET
    rth["date"] = pd.to_datetime(rth.index.date)
    sess = rth.groupby("date").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), n=("close", "size"))
    sess = sess[sess.n >= min_bars]
    # Parkinson gives a sigma in LOG units; multiply by the price level for points
    sess["sigma_pts"] = np.sqrt(np.log(sess.high / sess.low) ** 2 / (4 * np.log(2))) * sess.close
    sess["i_high"] = rth.groupby("date").apply(lambda x: int(np.argmax(x.high.values))).reindex(sess.index)
    sess["i_low"] = rth.groupby("date").apply(lambda x: int(np.argmin(x.low.values))).reindex(sess.index)
    return sess


def main() -> None:
    for sym in ("ES=F", "NQ=F"):
        s = sessions_1m(sym)
        R = (s.high - s.open) / s.sigma_pts
        rng = (s.high - s.low) / s.sigma_pts
        frac = s.i_high / 390.0
        print(f"{sym}: {len(s)} RTH sessions {s.index.min().date()} -> {s.index.max().date()}")
        print(f"   E[R+/sigma] = {R.mean():.3f}   theory {THEO_RUP:.3f}")
        print(f"   E[range/sigma] = {rng.mean():.3f}   theory {THEO_RANGE:.3f}")
        print(f"   median sigma = {s.sigma_pts.median():.1f} pts, bars/session = {int(s.n.median())}")
        print(f"   high in the first hour {np.mean(frac <= 1/6.5):.1%} "
              f"(arcsine 20.5%) | last hour {np.mean(frac >= 5.5/6.5):.1%}")


if __name__ == "__main__":
    main()
