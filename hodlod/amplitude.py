"""
A/B. Amplitude: expected O->H and O->L distances.

Convention: we work in units of the day's sigma.
    R+ = (H - O) / sigma_day,   R- = (O - L) / sigma_day,   A = R+ - R-
A is the asymmetry: it says whether the day is "up" or "down", and it is the only
one of the three that carries exploitable directional information.

Exact benchmark (driftless BM, horizon T, vol sigma):
    H - O =_d |N(0, sigma^2 T)|   =>   P(R+ <= x) = 2 Phi(x) - 1
because P(max < a) = 2Phi(a/s)-1. Hence E[R+] = sqrt(2/pi) = 0.7979 and
q_90(R+) = Phi^{-1}(0.95) = 1.6449. Any systematic deviation from this law is a
tail/jump/drift premium to be measured, not assumed.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto, norm


# ----------------------------------------------------------------------------
# A5. Half-normal benchmark
# ----------------------------------------------------------------------------
def halfnormal_cdf(x, sigma_rem=1.0):
    x = np.asarray(x, float)
    return 2.0 * norm.cdf(x / max(sigma_rem, 1e-12)) - 1.0


def halfnormal_quantile(q, sigma_rem=1.0):
    """q_alpha of (H-O) under a driftless BM: sigma * Phi^{-1}((1+alpha)/2)."""
    q = np.asarray(q, float)
    return sigma_rem * norm.ppf(0.5 + 0.5 * q)


def amplitude_multiplier(x, sigma_rem=1.0, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    """Empirical quantile / theoretical quantile of R+ for each probability level.

    > 1 everywhere => fatter tails than the BM (jumps, stochastic vol, drift).
    The shape of the ratio across alpha is the most informative diagnostic:
      - flat ratio > 1  : a simple scaling (mis-estimated vol),
      - increasing ratio: fat tails (jumps),
      - ratio > 1 at both ends: monitoring bounds + jumps.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    emp = np.quantile(x, qs)
    th = halfnormal_quantile(np.array(qs), sigma_rem)
    return dict(zip([f"q{int(q*100)}" for q in qs], (emp / th).tolist()))


# ----------------------------------------------------------------------------
# A6. EVT: tails of the normalised amplitude
# ----------------------------------------------------------------------------
def fit_gpd_tail(x, threshold_q: float = 0.90) -> dict:
    """GPD fit by peaks-over-threshold above the threshold_q quantile.

    Model: for u = q_F(x) and Y = x - u | x > u,  Y ~ GPD(xi, sigma).
    Output: dictionary with u, xi (shape = tail index), sigma (scale), and counts.
    xi > 0 => polynomial tail, xi = 0 => exponential.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = float(np.quantile(x, threshold_q))
    exc = x[x > u] - u
    if len(exc) < 50:
        raise ValueError("too few excesses to fit a GPD (<50)")
    xi, loc, scale = genpareto.fit(exc, floc=0.0)
    return {"u": u, "xi": float(xi), "sigma": float(scale),
            "n": int(len(x)), "n_excess": int(len(exc)),
            "threshold_q": float(threshold_q)}


def gpd_quantile(q, fit: dict) -> float:
    """POT quantile beyond the threshold: u + (sigma/xi)[((1-q)/(1-F_u))^{-xi} - 1]."""
    u, xi, sig = fit["u"], fit["xi"], fit["sigma"]
    Fu = fit["threshold_q"]
    q = float(q)
    if q <= Fu:
        raise ValueError("q must exceed the threshold quantile")
    if abs(xi) < 1e-8:
        return float(u - sig * np.log((1.0 - q) / (1.0 - Fu)))
    return float(u + (sig / xi) * (((1.0 - q) / (1.0 - Fu)) ** (-xi) - 1.0))


def gpd_tail_prob(x, fit: dict) -> float:
    """P(X > x) under the fitted GPD (x > u)."""
    u, xi, sig = fit["u"], fit["xi"], fit["sigma"]
    Fu = fit["threshold_q"]
    z = max(x - u, 0.0)
    if abs(xi) < 1e-8:
        return float((1 - Fu) * np.exp(-z / sig))
    return float((1 - Fu) * max(1.0 + xi * z / sig, 1e-12) ** (-1.0 / xi))


# ----------------------------------------------------------------------------
# A7. Quantile regression and pinball boosting
# ----------------------------------------------------------------------------
def pinball(y, q_hat, q):
    """Pinball (asymmetric absolute) loss: the only proper loss for a quantile."""
    y = np.asarray(y, float)
    e = y - np.asarray(q_hat, float)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def fit_quantreg(X, y, q: float = 0.5):
    """Linear quantile regression (Koenker & Bassett 1978) via statsmodels."""
    import statsmodels.api as sm
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return sm.QuantReg(np.asarray(y, float), sm.add_constant(X)).fit(q=q)


class QuantileBooster:
    """Gradient boosting with pinball loss (sklearn, loss='quantile').

    What it estimates: q_alpha(R+ | X), the conditional quantile of the amplitude,
    not the mean. Advantage over linear regression: interactions
    (vol regime x announcement day x gap), non-linearities, robustness to outliers.
    Risk: fast overfitting with few days; keep max_depth <= 3 and use early
    stopping on a temporal validation fold.
    """

    def __init__(self, q: float = 0.5, **kw):
        from sklearn.ensemble import GradientBoostingRegressor
        self.q = q
        self.model = GradientBoostingRegressor(
            loss="quantile", alpha=q,
            n_estimators=kw.get("n_estimators", 300),
            learning_rate=kw.get("learning_rate", 0.03),
            max_depth=kw.get("max_depth", 3),
            subsample=kw.get("subsample", 0.8),
            min_samples_leaf=kw.get("min_samples_leaf", 40),
            random_state=kw.get("random_state", 0),
        )

    def fit(self, X, y):
        self.model.fit(np.asarray(X, float), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model.predict(np.asarray(X, float))


def crossing_check(q_preds: dict) -> np.ndarray:
    """A set of quantiles must be monotone in alpha. Counts crossings.

    q_preds: {alpha: predictions}. A non-zero crossing rate signals an unstable
    model (frequent with forests / independent per-quantile boosting).
    """
    alphas = sorted(q_preds)
    P = np.column_stack([q_preds[a] for a in alphas])
    return np.mean(np.diff(P, axis=1) < 0, axis=1)


# ----------------------------------------------------------------------------
# B4. Asymmetry classifier: "which side extends?"
# ----------------------------------------------------------------------------
def asym_target(R_plus, R_minus):
    """1 if the upper branch dominates, 0 otherwise. Under a driftless BM: P = 1/2."""
    return (np.asarray(R_plus, float) > np.asarray(R_minus, float)).astype(int)


def fit_asym_classifier(X, y):
    """Calibrated classifier (gradient boosting + isotonic calibration)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    base = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                          learning_rate=0.05, l2_regularization=1.0,
                                          random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(np.asarray(X, float), np.asarray(y, int))
    return clf


def amplitude_features(df) -> np.ndarray:
    """Pre-open features for R+/R- (no data from day J).

    df must contain, for each day: rv_prev (1,5,22), var_pk_prev, range_prev_sigma,
    gap, overnight_range, vix, vix_chg, dow, is_event, close_pos_in_range_prev.
    """
    cols = ["log_rv1", "log_rv5", "log_rv22", "log_pk_prev", "range_prev_sigma",
            "gap_sigma", "on_range_sigma", "vix", "vix_chg", "dow", "is_event",
            "close_pos_prev"]
    return df[[c for c in cols if c in df.columns]].to_numpy(float)
