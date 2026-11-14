"""Reproducible FinzData access check. Never prints or stores the key."""
from __future__ import annotations
import numpy as np, pandas as pd
from research.providers import FinzDataProvider

def main():
    p = FinzDataProvider()
    print("armed:", p.armed)
    try:
        print("roots:", len(p.roots()))
    except Exception as e:
        print("roots failed:", type(e).__name__, e)
    for root in ("ES", "NQ"):
        try:
            c = p.contracts(root)
            print(f"{root} contracts: {len(c)} sample={list(c)[:4]}")
        except Exception as e:
            print(f"{root} contracts failed:", type(e).__name__, str(e)[:120])
        try:
            end = pd.Timestamp.now(tz="UTC").normalize()
            start = end - pd.Timedelta(days=30)
            df = p.bars(root, start.date().isoformat(), end.date().isoformat(), interval="1day")
            print(f"{root} daily: {len(df)} bars {df.ts.min()} -> {df.ts.max()} "
                  f"oi={df.open_interest.notna().any()} vol={df.volume.notna().any()}")
            print(df.tail(2)[["ts", "open", "high", "low", "close", "volume"]].to_string(index=False))
        except Exception as e:
            print(f"{root} daily failed:", type(e).__name__, str(e)[:160])

if __name__ == "__main__":
    main()
