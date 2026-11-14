"""
Volatility measurement and forecasting for the intraday HOD/LOD model.

Two distinct objects, often confused:
  - ESTIMATORS of the day's variance from OHLC -> Parkinson, Garman-Klass,
    Rogers-Satchell, Yang-Zhang. These are MEASUREMENT instruments, not forecasts.
  - FORECASTERS of day J's volatility from J-1..J-n -> HAR-RV, CARR, HEAVY.

The second feeds on the first: we measure RV properly (range estimator) to forecast
better (HAR), and we forecast in order to normalise amplitudes (section A).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

LN2 = np.log(2.0)


# ----------------------------------------------------------------------------
# A1. Variance estimators from OHLC
# ----------------------------------------------------------------------------
def parkinson_var(h, l):
    """Parkinson (1980). No drift, no gap. Var(range) = 4 ln2 * sigma^2 * T."""
    h = np.asarray(h, float)
    l = np.asarray(l, float)
    return np.log(h / l) ** 2 / (4.0 * LN2)


def garman_klass_var(o, h, l, c):
    """Garman & Klass (1980), closed form. No drift, no gap."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    return 0.5 * np.log(h / l) ** 2 - (2.0 * LN2 - 1.0) * np.log(c / o) ** 2


def rogers_satchell_var(o, h, l, c):
    """Rogers & Satchell (1991). Allows drift, but no opening gap."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    return np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)


def yang_zhang_var(o, h, l, c, prev_c):
    """Yang & Zhang (2000). Handles the opening gap (overnight component).

    sigma2_YZ = sigma2_overnight + k * sigma2_open-to-close + (1-k) * sigma2_RS
    k = 0.34 / (1.34 + (n+1)/(n-1))   -> weight of the open-to-close term
    Returns an average variance over the window (n >= 2).
    """
    o, h, l, c, prev_c = (np.asarray(x, float) for x in (o, h, l, c, prev_c))
    n = len(o)
    if n < 2:
        raise ValueError("Yang-Zhang requires at least 2 days")
    r_on = np.log(o / prev_c)              # overnight return
    r_oc = np.log(c / o)                   # open-to-close return
    var_on = np.var(r_on, ddof=1)
    var_oc = np.var(r_oc, ddof=1)
    var_rs = np.mean(rogers_satchell_var(o, h, l, c))
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    return var_on + k * var_oc + (1.0 - k) * var_rs


def ohlc_variance_table(df: pd.DataFrame) -> pd.DataFrame:
    """df: columns open/high/low/close, index = dates. Adds the 4 estimators."""
    out = df.copy()
    out["var_pk"] = parkinson_var(df.high, df.low)
    out["var_gk"] = garman_klass_var(df.open, df.high, df.low, df.close)
    out["var_rs"] = rogers_satchell_var(df.open, df.high, df.low, df.close)
    out["var_rv_1m"] = np.nan  # to be filled with RV computed on 1-minute bars
    out["var_cc"] = np.log(df.close / df.close.shift(1)) ** 2
    return out


def efficiency_gain() -> dict:
    """Asymptotic efficiency relative to the close-to-close estimator (=1).

    Values from the literature; they assume a driftless Brownian motion with no
    gap and continuous observation. On a real market they are upper bounds.
    """
    return {"close-to-close": 1.0, "parkinson": 5.2, "garman-klass": 7.4,
            "rogers-satchell": 8.0, "yang-zhang": 14.0}


# ----------------------------------------------------------------------------
# A2. Intraday seasonal profile and volatility time
# ----------------------------------------------------------------------------
def seasonal_profile(returns_by_bucket: np.ndarray, weekdays: np.ndarray | None = None,
                     smooth: int = 3) -> np.ndarray:
    """Normalised intraday variance profile (sum = 1).

    returns_by_bucket: shape (n_days, n_buckets), returns per bucket (1 or 5 minutes).
    Steps (Andersen-Bollerslev 1998 style):
      1. centre |r| by day (remove the day's volatility level),
      2. correct the weekday effect,
      3. cross-sectional average, smooth, normalise.
    The result s(t) satisfies sum_b s_b = 1 and serves as a variance density:
    F(t) = cumsum(s) is the fraction of variance elapsed = "volatility time".
    """
    r = np.asarray(returns_by_bucket, float)
    r = np.where(np.isfinite(r), r, np.nan)
    a = np.abs(r)
    # 1. day level
    daily = np.nanmean(a, axis=1, keepdims=True)
    a = a / np.where(daily > 0, daily, np.nan)
    # 2. weekday effect
    if weekdays is not None:
        wd = np.asarray(weekdays)
        for d in np.unique(wd):
            m = np.nanmean(a[wd == d])
            if np.isfinite(m) and m > 0:
                a[wd == d] = a[wd == d] / m
    # 3. cross-sectional average + smoothing
    prof = np.nanmean(a ** 2, axis=0)          # average variance per bucket
    if smooth > 1:
        k = np.ones(smooth) / smooth
        prof = np.convolve(np.r_[prof[smooth - 1:0:-1], prof, prof[-2:-smooth - 1:-1]],
                           k, mode="valid")
    prof = np.where(prof > 0, prof, np.nanmedian(prof))
    return prof / prof.sum()


def vol_time(profile: np.ndarray) -> np.ndarray:
    """F(t): fraction of cumulative variance. F(0)=0, F(1)=1."""
    return np.cumsum(np.asarray(profile, float) / np.sum(profile))


def seasonal_adjust(returns_by_bucket: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Deseasonalised returns: r / s(t). Used for dependence tests."""
    s = np.sqrt(np.asarray(profile, float))
    return np.asarray(returns_by_bucket, float) / np.where(s > 0, s, np.nan)


# ----------------------------------------------------------------------------
# A3. HAR-RV (Corsi 2009)
# ----------------------------------------------------------------------------
def _har_design(x: np.ndarray, lags=(1, 5, 22)) -> np.ndarray:
    """x: log(RV) aligned on t. Returns columns [RV_t, mean_5, mean_22]."""
    s = pd.Series(x)
    cols = [s.shift(0)]
    for k in lags[1:]:
        cols.append(s.rolling(k).mean())
    return np.column_stack([c.values for c in cols])


class HAR:
    """HAR-RV on log(RV). RV_{t+1} = c + b1 RV_t + b2 RVbar_5 + b3 RVbar_22 + e.

    Taking logs stabilises heteroscedasticity; the transform back to variance
    requires the smearing correction exp(sigma_hat^2 / 2).

    Assumptions: RV is an unbiased estimator of integrated variance (no
    microstructure noise, no poorly captured jumps); coefficients are stable.
    Where it breaks: volatility regime changes (2020, 2022), microstructure bias
    on 1-minute bars, and above all a missing previous-day RV on half sessions.
    """

    def __init__(self, lags=(1, 5, 22), log: bool = True, eps: float = 1e-12):
        self.lags, self.log, self.eps = lags, log, eps
        self.coef_ = None
        self.smear_ = 1.0
        self.r2_ = np.nan

    def _x(self, rv):
        rv = np.asarray(rv, float)
        return np.log(rv + self.eps) if self.log else rv

    def fit(self, rv, start: int | None = None):
        x = self._x(rv)
        X = _har_design(x, self.lags)
        y = np.r_[x[1:], np.nan]
        m = np.isfinite(X).all(1) & np.isfinite(y)
        if start is not None:
            m[:start] = False
        A = np.column_stack([np.ones(m.sum()), X[m]])
        self.coef_, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        resid = y[m] - A @ self.coef_
        self.smear_ = float(np.exp(np.var(resid, ddof=1) / 2.0)) if self.log else 1.0
        self.r2_ = float(1.0 - np.var(resid, ddof=1) / np.var(y[m], ddof=1))
        return self

    def predict(self, rv) -> np.ndarray:
        """Forecast of RV_{t+1} for each t (NaN when history is insufficient)."""
        x = self._x(rv)
        X = _har_design(x, self.lags)
        out = np.full(len(x), np.nan)
        m = np.isfinite(X).all(1)
        out[m] = np.column_stack([np.ones(m.sum()), X[m]]) @ self.coef_
        if self.log:
            out = np.exp(out) * self.smear_
        return out


# ----------------------------------------------------------------------------
# A4. CARR(1,1) (Chou 2005)
# ----------------------------------------------------------------------------
def carr_filter(R: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """lambda_t = omega + alpha * R_{t-1} + beta * lambda_{t-1}  (lambda_0 = omega)."""
    R = np.asarray(R, float)
    x = np.empty(len(R))
    x[0] = omega
    x[1:] = omega + alpha * R[:-1]
    return lfilter([1.0], [1.0, -beta], x)


def carr_nll(theta, R: np.ndarray, burn: int = 20, dist: str = "exp") -> float:
    """CARR quasi-ML negative log-likelihood.

    R_t = lambda_t * eps_t,  eps_t iid with mean 1.
    - dist='exp'      : eps ~ Exp(1)   -> nll = sum(log lambda + R/lambda)
    - dist='weibull'  : eps ~ Weibull(k) with mean 1 -> nll = sum(log lambda - log f_eps(R/lambda))
    """
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
        return 1e12
    lam = carr_filter(R, omega, alpha, beta)
    lam = lam[burn:]
    r = np.asarray(R, float)[burn:]
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        return 1e12
    if dist != "exp":
        raise NotImplementedError(
            "Only the exponential likelihood is implemented. "
            "For a Weibull with mean 1, replace the density below."
        )
    # eps ~ Exp(1): f(r|lam) = (1/lam) exp(-r/lam)
    return float(np.sum(np.log(lam) + r / lam))


class CARR:
    """CARR(1,1): Chou (2005), JMCB 37(3), 561-582.

    What it estimates exactly: lambda_t = E[R_t | F_{t-1}], where R_t = H_t - L_t
    (daily range), with R_t = lambda_t * eps_t and eps of mean 1. This is NOT a
    variance forecast: it is a forecast of the AMPLITUDE of the range.
    The full conditional distribution follows: P(R_t <= r | F) = F_eps(r/lambda_t).
    For eps ~ Exp(1): q_alpha(R_t) = -lambda_t * ln(1-alpha).

    Calibration: quasi-ML (the estimators remain consistent if eps is not
    exponential, provided E[eps]=1). Optimise in log parameters or with bounds;
    the constraint alpha+beta<1 guarantees stationarity.

    Assumptions: range is a good proxy for local volatility (Parkinson 1980); the
    dynamics are linear in R (no threshold, no asymmetry).
    Where it breaks: gap/event days (the range explodes without lambda_t
    anticipating it), slow regime changes (beta close to 1 => under-reaction),
    and the non-linearity documented by Chen, Gerlach & Lin (2008), who propose a
    threshold version (threshold CARR, Bayesian estimation).
    """

    def __init__(self, dist: str = "exp", burn: int = 20):
        self.dist, self.burn = dist, burn
        self.theta_ = None
        self.se_ = None

    def fit(self, R, x0=(0.5, 0.1, 0.8)):
        from scipy.optimize import minimize
        R = np.asarray(R, float)
        R = R[np.isfinite(R)]
        res = minimize(carr_nll, np.array(x0, float), args=(R, self.burn, self.dist),
                       method="L-BFGS-B",
                       bounds=[(1e-8, None), (0.0, 0.999), (0.0, 0.999)])
        if not res.success and res.fun >= 1e11:
            raise RuntimeError(f"CARR: optimisation failed ({res.message})")
        self.theta_ = res.x
        self.se_ = self._std_errors(R, res)
        return self

    def _std_errors(self, R, res):
        try:
            h = 1e-4
            g = np.zeros((len(R) - self.burn, 3))
            for i in range(3):
                dp = np.zeros(3)
                dp[i] = h
                g[:, i] = (carr_nll(res.x + dp, R, self.burn) -
                           carr_nll(res.x - dp, R, self.burn)) / (2 * h)
            # BHHH approximation: sum of individual gradients
            H = g.T @ g
            cov = np.linalg.pinv(H)
            return np.sqrt(np.clip(np.diag(cov), 0, None))
        except Exception:
            return np.full(3, np.nan)

    @property
    def persistence(self) -> float:
        return float(self.theta_[1] + self.theta_[2])

    def filter(self, R) -> np.ndarray:
        return carr_filter(np.asarray(R, float), *self.theta_)

    def predict_next(self, R) -> float:
        o, a, b = self.theta_
        lam = carr_filter(np.asarray(R, float), o, a, b)[-1]
        return float(o + a * np.asarray(R, float)[-1] + b * lam)

    @staticmethod
    def quantile(lam, q, dist: str = "exp"):
        lam = np.asarray(lam, float)
        if dist == "exp":
            return lam * (-np.log1p(-np.asarray(q, float)))
        raise NotImplementedError("implement the chosen law for eps")


def realized_variance_1m(closes: np.ndarray, open_price: float | None = None) -> float:
    """RV = sum of r^2 on 1-minute bars (in logs). Serves as label for HAR/CARR."""
    c = np.asarray(closes, float)
    if open_price is not None:
        c = np.r_[open_price, c]
    r = np.diff(np.log(c))
    return float(np.sum(r ** 2))


def bipower_variation(closes: np.ndarray, open_price: float | None = None) -> float:
    """BV = (pi/2) * sum |r_t| |r_{t-1}|. Jump-robust estimator
    (Barndorff-Nielsen & Shephard 2006). RV - BV approximates the jump component."""
    c = np.asarray(closes, float)
    if open_price is not None:
        c = np.r_[open_price, c]
    r = np.diff(np.log(c))
    return float((np.pi / 2.0) * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
