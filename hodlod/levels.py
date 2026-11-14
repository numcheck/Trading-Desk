"""
B5. Confluence levels: PDH/PDL, overnight high/low, VWAP, initial balance,
volume profile, round numbers.

The central trap: "the PDH is touched 70% of the time" means nothing. A level
close to the price is touched more often, mechanically. So we must:
  1) condition on the distance in sigma,
  2) compare with a PLACEBO level at the same distance,
  3) define "rejection" and "break" symmetrically and without look-ahead.

This module provides all three. The frequencies themselves must be estimated on
YOUR data: they are not invented here.
"""
from __future__ import annotations

import numpy as np


def reference_levels(prev: dict, overnight: dict, session: dict | None = None) -> dict:
    """Levels known BEFORE the RTH open.

    prev     : {'high','low','close','vwap'} of the previous RTH session
    overnight: {'high','low','close','vwap'} of the ETH session (16:00->09:30)
    """
    lv = {
        "pdh": prev.get("high"), "pdl": prev.get("low"), "pdc": prev.get("close"),
        "pd_vwap": prev.get("vwap"),
        "onh": overnight.get("high"), "onl": overnight.get("low"),
        "onc": overnight.get("close"), "on_vwap": overnight.get("vwap"),
    }
    if session:
        lv.update({f"s_{k}": v for k, v in session.items()})
    return {k: float(v) for k, v in lv.items() if v is not None and np.isfinite(v)}


def initial_balance(bars, minutes: int = 30) -> dict:
    """Initial balance: high/low of the first `minutes` minutes. Known at 10:00."""
    ib = bars.iloc[:minutes]
    return {"ibh": float(ib.high.max()), "ibl": float(ib.low.min())}


def touch_break(highs, lows, closes, level: float, tol: float,
                side: str = "up") -> dict:
    """Event "touched then rejected" vs "touched then broken", without look-ahead.

    side='up'  : we test a level above the price (PDH, ONH, IBH).
    A TOUCH is a wick inside [level-tol, level+tol].
    A BREAK is a close beyond level+tol (or level-tol if side='down').
    A REJECTION is a touch with no break and a return beyond tol to the origin side.
    """
    h = np.asarray(highs, float)
    l = np.asarray(lows, float)
    c = np.asarray(closes, float)
    n = len(h)
    if side == "up":
        hit = np.where(h >= level - tol)[0]
    else:
        hit = np.where(l <= level + tol)[0]
    if len(hit) == 0:
        return {"touched": 0, "broke": 0, "rejected": 0, "first_hit": np.nan}
    i0 = int(hit[0])
    if side == "up":
        broke = bool(np.any(c[i0:] > level + tol))
        rejected = bool(not broke and np.any(c[i0:] < level - 3 * tol))
    else:
        broke = bool(np.any(c[i0:] < level - tol))
        rejected = bool(not broke and np.any(c[i0:] > level + 3 * tol))
    return {"touched": 1, "broke": int(broke), "rejected": int(rejected),
            "first_hit": i0 / max(n - 1, 1)}


def placebo_levels(open_px: float, sigma_day: float, distances_sigma: np.ndarray,
                   n: int, seed: int = 0, side: str = "up") -> np.ndarray:
    """Fake levels at distances drawn from the real distribution.

    This is the indispensable control: the rejection rate of a REAL level is
    compared with that of a meaningless level placed at the SAME distance.
    If the difference is not significant, the "level" adds nothing.
    """
    rng = np.random.default_rng(seed)
    d = rng.choice(np.asarray(distances_sigma, float), size=n, replace=True)
    d = d * sigma_day
    return open_px + d if side == "up" else open_px - d


def confluence_table(days: list[dict], level_key: str, tol_sigma: float = 0.05,
                     placebo: bool = True, n_boot: int = 2000, seed: int = 0) -> dict:
    """Touch/break/rejection frequency table, with bootstrap CI and placebo.

    days: list of dicts, one per session, containing at least
          {'highs','lows','closes','open','sigma_day','levels'} where 'levels' is
          the dict produced by reference_levels().
    Returns the real rates, the placebo rates and the difference with a 95% CI.
    That difference is the only interpretable number.
    """
    real, plac, dists = [], [], []
    for d in days:
        lv = d["levels"].get(level_key)
        if lv is None or not np.isfinite(lv):
            continue
        tolv = tol_sigma * d["sigma_day"]
        side = "up" if lv >= d["open"] else "down"
        r = touch_break(d["highs"], d["lows"], d["closes"], lv, tolv, side)
        if not r["touched"]:
            continue
        real.append(r["rejected"])
        dists.append(abs(lv - d["open"]) / max(d["sigma_day"], 1e-9))
        if placebo:
            open_px = float(d["open"])
            sd = float(d["sigma_day"])
            rng = np.random.default_rng(seed + len(real))
            fake = placebo_levels(open_px, sd, np.array(dists), 1, int(rng.integers(1e9)), side)[0]
            rp = touch_break(d["highs"], d["lows"], d["closes"], fake, tolv, side)
            plac.append(rp["rejected"] if rp["touched"] else 0)
    real = np.array(real, float)
    if len(real) == 0:
        return {"n": 0}
    out = {"n": int(len(real)), "reject_rate": float(real.mean()),
           "reject_rate_se": float(real.std(ddof=1) / np.sqrt(len(real))) if len(real) > 1 else np.nan}
    if placebo and plac:
        plac = np.array(plac, float)
        diff = real.mean() - plac.mean()
        rng = np.random.default_rng(seed)
        bs = np.array([rng.choice(real, len(real)).mean() - rng.choice(plac, len(plac)).mean()
                       for _ in range(n_boot)])
        out.update({"placebo_reject_rate": float(plac.mean()),
                    "delta": float(diff),
                    "delta_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    "significant": bool(np.quantile(bs, 0.025) > 0 or np.quantile(bs, 0.975) < 0)})
    out["mean_distance_sigma"] = float(np.mean(dists))
    return out


def volume_profile_poc(bars, bin_size: float = 0.25, value_area: float = 0.70) -> dict:
    """Point of Control and value area approximated by a volume histogram.

    Honest approximation: without tick-by-tick data, each bar's volume is spread
    uniformly over its range. Biased but usable for a confluence level; do not
    present it as a true volume profile.
    """
    lo = np.floor(bars.low.min() / bin_size) * bin_size
    hi = np.ceil(bars.high.max() / bin_size) * bin_size
    edges = np.arange(lo, hi + bin_size, bin_size)
    if len(edges) < 3:
        return {"poc": float(bars.close.mean()), "val": float(bars.low.min()),
                "vah": float(bars.high.max())}
    vol = np.zeros(len(edges) - 1)
    for _, b in bars.iterrows():
        plo, phi = b.low, b.high
        m = (edges[:-1] < phi) & (edges[1:] > plo)
        k = m.sum()
        if k:
            vol[m] += b.volume / k
    poc_i = int(np.argmax(vol))
    order = np.argsort(vol)[::-1]
    cum, sel = 0.0, []
    for i in order:
        sel.append(i)
        cum += vol[i]
        if cum >= value_area * vol.sum():
            break
    sel = np.sort(sel)
    return {"poc": float(0.5 * (edges[poc_i] + edges[poc_i + 1])),
            "val": float(edges[sel[0]]), "vah": float(edges[sel[-1] + 1])}
