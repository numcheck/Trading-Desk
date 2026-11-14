"""Fast, network-free tests of the backtest guards.

Each test pins a specific way the engine could lie: a leaked future value, a
duplicated bet, a mispriced trade, an optimistic fill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hodlod.evaluate import leakage_audit
from hodlod.realtime import session_features
from research.engine import (COST_SIGMA, INSTRUMENTS, cost_scenarios,
                             har_walkforward, pick, simulate)


# --------------------------------------------------------------------- look-ahead
def test_har_is_prefix_invariant():
    """The forecast for day t must not change when later days are appended.

    If it does, the expanding-window fit is seeing the future.
    """
    rng = np.random.default_rng(0)
    var = rng.gamma(2.0, 1e-4, 360)
    full = har_walkforward(var)
    for cut in (160, 220, 300):
        part = har_walkforward(var[:cut])
        np.testing.assert_allclose(part[:cut], full[:cut], rtol=1e-12, atol=0.0,
                                   equal_nan=True)


def test_session_features_are_prefix_invariant():
    """Features at bar i must be identical whether or not later bars exist."""
    rng = np.random.default_rng(1)
    n = 60
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    bars = pd.DataFrame({
        "open": price, "close": price + rng.standard_normal(n) * 0.02,
        "high": price + 0.1, "low": price - 0.1,
        "volume": rng.integers(100, 1000, n).astype(float)})
    F = np.linspace(0.0, 1.0, n)
    for i in (10, 25, 40):
        a = session_features(bars, i, sigma_day=1.0, F_grid=F)
        b = session_features(bars.iloc[: i + 1], i, sigma_day=1.0, F_grid=F[: i + 1])
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-12, abs=1e-12), k


def test_leakage_audit_flags_suspicious_names():
    assert leakage_audit(["sigma", "F_now"])["ok"]
    bad = leakage_audit(["hod_final", "sigma"])["suspect_features"]
    assert bad == ["hod_final"]


# --------------------------------------------------------------------- concurrency
def _cand() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05"] * 4 + ["2026-01-06"] * 3),
        "slot": [0, 1, 2, 3, 0, 1, 2],
        "side": ["short", "long", "short", "long", "short", "short", "long"],
        "p": [0.90, 0.85, 0.95, 0.70, 0.60, 0.90, 0.50],
        "g_1.0": [1.0] * 7, "n_1.0": [0.9] * 7, "r_1.0": ["tp"] * 7,
        "cost_R": [0.02] * 7,
    })


def test_one_trade_per_day_by_default():
    out = pick(_cand(), theta=0.5, kt=1.0)
    assert out.groupby("date").size().max() == 1
    assert len(out) == 2
    # the first valid signal of each day, by (slot, side)
    assert out.loc[out.date == pd.Timestamp("2026-01-05"), "slot"].item() == 0
    assert out.loc[out.date == pd.Timestamp("2026-01-06"), "slot"].item() == 0


def test_day_cap_is_configurable():
    assert len(pick(_cand(), theta=0.5, kt=1.0, max_per_day=2)) == 4
    assert len(pick(_cand(), theta=0.5, kt=1.0, max_per_day=0)) == 7


# --------------------------------------------------------------------- execution
def _bars() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
        "slot": [0, 1],
        "open": [100.0, 100.0], "high": [100.5, 101.5],
        "low": [99.5, 98.0], "close": [100.0, 99.0], "volume": [1.0, 1.0],
    })


def test_stop_wins_when_both_touched_in_the_same_bar():
    """A bar that touches both levels must resolve as a stop, never a take-profit."""
    rt = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "slot": [0],
                       "sigma": [1.0], "p_hi": [1.0], "p_lo": [0.0]})
    out = simulate(rt, _bars(), [1.0])
    short = out[(out.side == "short")].iloc[0]
    assert short["r_1.0"] == "stop"
    assert short["g_1.0"] < 0


def test_risk_filter_skips_wide_stops():
    """Trades whose stop would exceed 0.7 sigma are not taken."""
    rt = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"]), "slot": [0],
                       "sigma": [0.1], "p_hi": [1.0], "p_lo": [0.0]})
    assert len(simulate(rt, _bars(), [1.0])) == 0


# --------------------------------------------------------------------- costs
def test_cost_scenarios_are_monotone_and_anchored():
    oos = pd.DataFrame({"gross": np.full(100, 0.05), "cost_R": np.full(100, COST_SIGMA),
                        "reason": ["tp"] * 50 + ["stop"] * 50,
                        "net": np.full(100, 0.05 - COST_SIGMA)})
    es = cost_scenarios(oos, "ES")
    nets = [r["net_mean_R"] for r in es]
    assert nets[0] > nets[1] > nets[2]
    # the 2-tick scenario must reproduce the base cost exactly: 0.5 pt on a 31 pt sigma
    assert es[1]["cost_R"] == pytest.approx(0.5 / INSTRUMENTS["ES"]["sigma_pts"])
    # stop exits carry one extra tick, take-profits do not
    assert es[1]["stop_slip_R"] > 0


def test_cheaper_instrument_costs_less():
    oos = pd.DataFrame({"gross": np.full(50, 0.05), "cost_R": np.full(50, COST_SIGMA),
                        "reason": ["tp"] * 25 + ["stop"] * 25, "net": np.zeros(50)})
    es = cost_scenarios(oos, "ES")[1]["cost_R"]
    nq = cost_scenarios(oos, "NQ")[1]["cost_R"]
    assert nq < es


# --------------------------------------------------------------------- calendar
def test_timezone_handles_dst():
    """09:30 ET is -5h in winter and -4h in summer: never a fixed UTC offset."""
    winter = pd.Timestamp("2026-01-15 09:30", tz="America/New_York")
    summer = pd.Timestamp("2026-07-15 09:30", tz="America/New_York")
    assert winter.utcoffset().total_seconds() == -5 * 3600
    assert summer.utcoffset().total_seconds() == -4 * 3600
