"""
C. Timing: at what time H and L form.

Reference result: for a driftless Brownian motion, the time of the maximum on
[0,1] follows the arcsine law (third arcsine law, Levy 1939/1948):

    P(T_H <= t) = (2/pi) * arcsin(sqrt(t)),   density 1/(pi sqrt(t(1-t)))

So it is U-shaped, with mass at both ends, and it is NOT an empirical property of
intraday volatility: it is a property of any continuous random walk. The seasonal
volatility profile adds a second layer: the relevant clock is no longer t but
F(t), the fraction of variance elapsed.

This module provides: the reference law, the transform to volatility time, the
diagnostics (PIT, KS, log-score) and a discrete-time hazard model that bridges
timing and real-time.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm


# ----------------------------------------------------------------------------
# C1. The arcsine law
# ----------------------------------------------------------------------------
def arcsine_cdf(t):
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(t))


def arcsine_pdf(t):
    t = np.clip(np.asarray(t, float), 1e-12, 1 - 1e-12)
    return 1.0 / (np.pi * np.sqrt(t * (1.0 - t)))


def arcsine_ppf(u):
    u = np.clip(np.asarray(u, float), 1e-12, 1 - 1e-12)
    return np.sin(np.pi * u / 2.0) ** 2


def arcsine_expected_mass(bins) -> np.ndarray:
    """Probability mass of the reference law on given intervals."""
    bins = np.asarray(bins, float)
    return np.diff(arcsine_cdf(bins))


# ----------------------------------------------------------------------------
# C2. Volatility time
# ----------------------------------------------------------------------------
def clock_to_vol_time(t, F_grid):
    """t as a fraction of the session -> F(t) as a fraction of variance."""
    grid = np.linspace(0.0, 1.0, len(F_grid))
    return np.interp(np.asarray(t, float), grid, np.asarray(F_grid, float))


def vol_time_to_clock(F, F_grid):
    grid = np.linspace(0.0, 1.0, len(F_grid))
    return np.interp(np.asarray(F, float), np.asarray(F_grid, float), grid)


def arcsine_cdf_voltime(t, F_grid):
    """Reference law when volatility is seasonal: arcsine in volatility time."""
    return arcsine_cdf(clock_to_vol_time(t, F_grid))


# ----------------------------------------------------------------------------
# C3. Diagnostics
# ----------------------------------------------------------------------------
def pit_from_argmax(argmax_frac, F_grid):
    """Two-step PIT: u = ArcsinCDF( F(t_max) ), then a uniformity test.

    Why two steps: under H0, F(t_max) follows an ARCSINE law (not a uniform one) —
    that is precisely Levy's result. So ArcsinCDF is applied on top. Forgetting
    this second step is the most common mistake in this diagnostic.

    Under H0 (BM + known vol profile), u ~ U(0,1).
    Inverted-U PIT => extremes more often in the middle than expected.
    Bell-shaped PIT => extremes earlier/later than the arcsine law.
    """
    return arcsine_cdf(clock_to_vol_time(np.asarray(argmax_frac, float), F_grid))


def pit_report(argmax_frac, F_grid, alpha: float = 0.05) -> dict:
    u = pit_from_argmax(argmax_frac, F_grid)
    u = u[np.isfinite(u)]
    ks = kstest(u, "uniform")
    return {
        "n": int(len(u)),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "mean_pit": float(np.mean(u)),
        "reject_uniform": bool(ks.pvalue < alpha),
        "note": "bell-shaped PIT => extremes earlier/later than the arcsine law",
    }


def pit_report_clock(argmax_frac, alpha: float = 0.05) -> dict:
    """Same diagnostic WITHOUT volatility time (raw clock).

    Serves as a negative control: if this version is not rejected while the
    volatility-time version is, the volatility profile is badly estimated.
    """
    u = arcsine_cdf(np.asarray(argmax_frac, float))
    u = u[np.isfinite(u)]
    ks = kstest(u, "uniform")
    return {"ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "reject_uniform": bool(ks.pvalue < alpha)}


def discrete_logscore(counts: np.ndarray, p: np.ndarray) -> float:
    """Log-score of a discrete distribution (smaller is better)."""
    counts = np.asarray(counts, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(counts * np.log(p)) / max(counts.sum(), 1.0))


def brier_multiclass(onehot: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((np.asarray(p) - np.asarray(onehot)) ** 2, axis=1)))


# ----------------------------------------------------------------------------
# C4. Discrete-time hazard (bridge to real time)
# ----------------------------------------------------------------------------
def empirical_hazard(argmax_frac, n_bins: int = 48, F_grid=None):
    """Empirical hazard h_k = P(first crossing in bin k | not crossed yet).

    Returns (bins, h, S) where S is the implied survival: S_k = prod_{j<=k}(1-h_j).
    This is the non-parametric version of p_final(t); it serves as a control for
    the diffusion model and as a target for a classifier.
    """
    a = np.asarray(argmax_frac, float)
    a = a[np.isfinite(a)]
    edges = np.linspace(0, 1, n_bins + 1)
    if F_grid is not None:
        # bin in volatility time, which is more interpretable
        a = clock_to_vol_time(a, F_grid)
    idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    at_risk = np.cumsum(cnt[::-1])[::-1]
    h = np.where(at_risk > 0, cnt / np.maximum(at_risk, 1e-9), 0.0)
    S = np.cumprod(1.0 - h)
    return edges, h, S


def competing_risk_table(argmax_high, argmin_low, n_bins: int = 24):
    """Competing-risks table: H and L compete for the session.

    Returns, per bin, the frequency of "HOD set in this bin", "LOD set in this bin",
    and the probability that BOTH are already set (the absorbing range state).
    Useful for a range-fade strategy rather than an extreme-reversal one.
    """
    hi = np.clip((np.asarray(argmax_high) * n_bins).astype(int), 0, n_bins - 1)
    lo = np.clip((np.asarray(argmin_low) * n_bins).astype(int), 0, n_bins - 1)
    n = len(hi)
    both = np.maximum(hi, lo)
    return {
        "bin_high": np.bincount(hi, minlength=n_bins) / max(n, 1),
        "bin_low": np.bincount(lo, minlength=n_bins) / max(n, 1),
        "bin_both_set": np.bincount(both, minlength=n_bins) / max(n, 1),
    }


def session_argmax_min(bars) -> tuple[float, float]:
    """Time (fraction of the session) of the high and of the low, from 1-minute bars."""
    h = np.asarray(bars.high.values, float)
    l = np.asarray(bars.low.values, float)
    n = len(h)
    return float(np.argmax(h) / max(n - 1, 1)), float(np.argmin(l) / max(n - 1, 1))
