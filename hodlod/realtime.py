"""
D. Real time: P(the running HOD/LOD is the final one | information at t).

The core object is the survival probability of a maximum. We provide:
  - the driftless formula (reflection principle),
  - its drift extension,
  - the two-barrier (range) case,
  - volatility time,
  - the feature builder and the calibration layer.

The crucial point: these formulas assume a Brownian motion with CONSTANT
volatility. The real market violates three assumptions (drift, fat tails/jumps,
stochastic volatility). The empirical calibration layer is what makes the object
usable; the formula is a benchmark the ML must beat, not a truth.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------
# D1. Closed forms
# ----------------------------------------------------------------------------
def p_no_new_high(d, s):
    """P(the running maximum is not exceeded over the remaining horizon), no drift.

    d: distance (in price) between the running max and the current price, d >= 0
    s: standard deviation of the price over the remaining horizon = sigma * sqrt(tau)

    Derivation: let M_t be the running max and X_t the price. For a driftless BM
    starting at X_t, P(max_{[t,T]} < M_t) = P(max_{u<=tau} (X_t + sigma W_u) < M_t)
    = 2*Phi(d/s) - 1, with d = M_t - X_t. (Reflection principle:
    P(M_tau >= a) = 2 P(B_tau >= a) => P(M_tau < a) = 1 - 2(1-Phi(a/s)) = 2Phi(a/s)-1.)

    This is exactly the formula in the brief. Limits: d/s -> 0 => p -> 0
    (at the high with the price at the high, a new high is nearly certain),
    large d/s => p -> 1 (but slowly).
    """
    d = np.asarray(d, float)
    s = np.maximum(np.asarray(s, float), 1e-12)
    return np.clip(2.0 * norm.cdf(d / s) - 1.0, 0.0, 1.0)


def p_no_new_high_drift(d, mu, s):
    """Same probability with drift.

    d : distance to the running maximum (price)
    mu: drift over the remaining horizon (price) = mu_rate * tau
    s : sigma * sqrt(tau)

    Absorbing-barrier formula:
      P(max < a) = Phi((a - mu)/s) - exp(2*mu*a/s^2) * Phi((-a - mu)/s)
    Check: mu -> 0 recovers 2Phi(a/s) - 1.
    """
    from scipy.special import log_ndtr
    d = np.atleast_1d(np.asarray(d, float))
    s = np.maximum(np.atleast_1d(np.asarray(s, float)), 1e-12)
    mu = np.broadcast_to(np.asarray(mu, float), d.shape).astype(float)
    # zero drift -> driftless formula (avoids numerical cancellation)
    zero = np.abs(mu) < 1e-12 * s
    mu_safe = np.where(zero, 0.0, mu)
    # exp(2 mu d / s^2) * Phi((-d-mu)/s) computed in logs to stay stable
    log_term = log_ndtr((-d - mu_safe) / s) + 2.0 * mu_safe * d / s ** 2
    general = norm.cdf((d - mu_safe) / s) - np.exp(np.clip(log_term, -700.0, 700.0))
    out = np.where(zero, 2.0 * norm.cdf(d / s) - 1.0, general)
    return np.clip(out, 0.0, 1.0)


def p_range_within(a, b, s, n_terms: int = 200):
    """P(-a < min and max < b) for a driftless BM, total stdev s, starting at 0.

    Image method with TWO absorbing barriers:
      sum_k [ Phi((b + 2kL)/s) - Phi((-a + 2kL)/s) ]
    - sum_k [ Phi((b + 2a + 2kL)/s) - Phi((a + 2kL)/s) ],   L = a + b.
    The second sum is essential: omitting it massively overstates the probability
    (0.68 instead of 0.37 for a=b=s). Verified by Monte-Carlo.

    Limit b -> inf: the second sum -> 1 - Phi(a/s) and the total -> 2Phi(a/s)-1,
    which recovers the one-barrier formula.
    Convergence is fast when min(a,b)/s >= 0.25 and slow below (keep n_terms high
    or switch to Monte-Carlo).
    """
    a = float(a)
    b = float(b)
    s = max(float(s), 1e-12)
    L = a + b
    k = np.arange(-n_terms, n_terms + 1, dtype=float)
    direct = norm.cdf((b + 2.0 * k * L) / s) - norm.cdf((-a + 2.0 * k * L) / s)
    reflect = norm.cdf((b + 2.0 * a + 2.0 * k * L) / s) - norm.cdf((a + 2.0 * k * L) / s)
    return float(np.sum(direct - reflect))


def remaining_sigma(sigma_day: float, F_now: float) -> float:
    """Sigma over the remaining horizon, in volatility time.

    F_now = fraction of variance already elapsed (0 at the open, 1 at the close).
    sigma_remaining = sigma_day * sqrt(1 - F_now).
    This is THE place where we gain the most versus clock time: the last half hour
    typically carries far more variance than a uniform average.
    """
    return float(sigma_day * np.sqrt(max(1.0 - F_now, 0.0)))


def p_final_now(price, running_max, running_min, sigma_day, F_now,
                side: str = "high", drift_bps: float = 0.0) -> float:
    """Convenience wrapper: probability that the running HOD (or LOD) is final."""
    s = remaining_sigma(sigma_day, F_now)
    if s <= 0:
        return 1.0
    if side == "high":
        d = max(running_max - price, 0.0)
        return float(p_no_new_high(d, s)) if drift_bps == 0 else \
            float(p_no_new_high_drift(np.array([d]), drift_bps * price * 1e-4, np.array([s]))[0])
    d = max(price - running_min, 0.0)
    if drift_bps == 0:
        return float(p_no_new_high(d, s))
    return float(p_no_new_high_drift(np.array([d]), -drift_bps * price * 1e-4, np.array([s]))[0])


# ----------------------------------------------------------------------------
# D2. Version averaged over stochastic volatility
# ----------------------------------------------------------------------------
def p_final_mixture(d, sigma_day, F_now, vol_of_vol: float = 0.35,
                    n_nodes: int = 9) -> float:
    """Integrates the formula over a log-normal distribution of sigma.

    Effect: mixing sigmas creates fatter tails, which INCREASES P(new high) far
    from the extreme -> p_final falls.
    vol_of_vol = standard deviation of log(sigma) (0.3-0.5 typical for indices).
    """
    s0 = remaining_sigma(sigma_day, F_now)
    if s0 <= 0:
        return 1.0
    z = np.linspace(-2.5, 2.5, n_nodes)
    w = norm.pdf(z)
    w = w / w.sum()
    sig = s0 * np.exp(vol_of_vol * z - 0.5 * vol_of_vol ** 2)
    return float(np.sum(w * p_no_new_high(np.full(n_nodes, d), sig)))


# ----------------------------------------------------------------------------
# D3. Real-time features (no look-ahead)
# ----------------------------------------------------------------------------
def session_features(bars, i: int, sigma_day: float, F_grid: np.ndarray,
                     ctx: dict | None = None) -> dict:
    """Features available at bar i (inclusive), for one session.

    bars: DataFrame indexed by minute with open/high/low/close/volume.
    F_grid: cumulative variance fraction, same length as bars.
    ctx: reference levels computed BEFORE the open (pdh, pdl, onh, onl, vwap...).

    Golden rule: every variable used here must be known at the end of bar i.
    The final HOD NEVER enters the features.
    """
    ctx = ctx or {}
    sub = bars.iloc[: i + 1]
    px = float(sub.close.iloc[-1])
    rmax = float(sub.high.max())
    rmin = float(sub.low.min())
    sd = max(float(sigma_day), 1e-9)
    F_now = float(F_grid[min(i, len(F_grid) - 1)])
    s_rem = remaining_sigma(sd, F_now)

    r1 = np.diff(np.log(np.r_[sub.open.iloc[0], sub.close.values]))
    rv_now = float(np.sum(r1 ** 2))
    F_elapsed = max(F_now, 1e-6)
    rv_expected = sd ** 2 * F_elapsed
    vol_ratio = rv_now / rv_expected if rv_expected > 0 else 1.0

    vwap = float(np.sum(sub.close * sub.volume) / max(sub.volume.sum(), 1e-9)) \
        if "volume" in sub else px

    feats = {
        "F_now": F_now,
        "d_high_sigma": (rmax - px) / max(s_rem, 1e-9),
        "d_low_sigma": (px - rmin) / max(s_rem, 1e-9),
        "p_high_diffusion": float(p_no_new_high(rmax - px, s_rem)),
        "p_low_diffusion": float(p_no_new_high(px - rmin, s_rem)),
        "p_high_mixture": p_final_mixture(rmax - px, sd, F_now),
        "vol_ratio": float(vol_ratio),
        "range_so_far_sigma": (rmax - rmin) / sd,
        "up_amp_sigma": (rmax - float(sub.open.iloc[0])) / sd,
        "dn_amp_sigma": (float(sub.open.iloc[0]) - rmin) / sd,
        "asym_amp": ((rmax - float(sub.open.iloc[0])) - (float(sub.open.iloc[0]) - rmin)) / sd,
        "dist_vwap_sigma": (px - vwap) / sd,
        "above_vwap": float(px > vwap),
        "log_vol": float(np.log1p(sub.volume.sum())) if "volume" in sub else 0.0,
        "trend_slope": float(np.polyfit(np.arange(len(sub)), np.log(sub.close.values), 1)[0] * 60.0)
        if len(sub) > 5 else 0.0,
        "minute_of_day": float(i),
    }
    for k, v in ctx.items():
        if isinstance(v, (int, float, np.floating)) and v is not None:
            feats[f"ctx_{k}"] = float(v)
            if "pdh" in k or "pdl" in k or "onh" in k or "onl" in k:
                feats[f"dist_{k}_sigma"] = (px - float(v)) / sd
    return feats


# ----------------------------------------------------------------------------
# D4. Calibration
# ----------------------------------------------------------------------------
class IsotonicCalibrator:
    """Isotonic calibration with linear extrapolation outside the fitted range.

    Why calibrate: the diffusion model is over-confident in calm regimes and
    under-confident after a shock. Calibration must be estimated on a separate
    sample (walk-forward), never on the classifier's training set.
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self._iso = None

    def fit(self, p, y):
        from sklearn.isotonic import IsotonicRegression
        self._iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                       out_of_bounds=self.out_of_bounds).fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self._iso.predict(np.asarray(p, float)), 0.0, 1.0)

    def slope(self, p, y):
        """Calibration slope: 1 = perfect, <1 = over-confident, >1 = under-confident."""
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        if p.std() == 0:
            return np.nan
        return float(np.polyfit(p, y, 1)[0])
