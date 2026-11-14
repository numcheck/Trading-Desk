"""
Validation: purged walk-forward, proper metrics, costs, overfitting.

Hard rules enforced here:
  - no look-ahead: the label of day J (final HOD) is only available at the close
    of J, so any model predicting an event of J trains on <= J-1.
  - purge: remove from training the days whose label overlaps the test window
    (useful when the target is a rolling average or an h-day horizon).
  - embargo: remove a few days before the test to break residual
    autocorrelation (Lopez de Prado 2018, ch. 7).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------
# Temporal splitting
# ----------------------------------------------------------------------------
def purged_walk_forward(n: int, train: int, test: int, embargo: int = 0,
                        label_horizon: int = 0):
    """Yields (idx_train, idx_test) moving forward in time, without overlap.

    label_horizon: number of days over which the training label "sees" the future.
    Training samples whose label spills into the test are removed.
    """
    start = train
    while start + test <= n:
        tr = np.arange(0, start)
        te = np.arange(start, start + test)
        if label_horizon > 0:
            tr = tr[tr + label_horizon < start]
        if embargo > 0:
            tr = tr[tr < start - embargo]
        yield tr, te
        start += test


def split_by_day(n_days: int, frac_train=0.6, frac_val=0.2):
    a = int(n_days * frac_train)
    b = int(n_days * (frac_train + frac_val))
    return np.arange(a), np.arange(a, b), np.arange(b, n_days)


# ----------------------------------------------------------------------------
# Probabilistic metrics
# ----------------------------------------------------------------------------
def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p, p_ref):
    """BSS = 1 - BS/BS_ref. > 0 means we beat the reference.

    The reference must be the analytic (diffusion) model, not climatology.
    Comparing with climatology is misleading: the base rate of "the running HOD
    is final" rises towards 1 late in the session, so climatology is already good.
    """
    b0 = brier(y, p_ref)
    return float(1.0 - brier(y, p) / b0) if b0 > 0 else np.nan


def logloss(y, p, eps: float = 1e-12):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y, p, n_bins: int = 10):
    """Reliability table: mean predicted probability vs observed frequency."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() > 0:
            rows.append({"bin": b, "n": int(m.sum()), "p_mean": float(p[m].mean()),
                         "y_mean": float(y[m].mean())})
    return rows


def expected_calibration_error(y, p, n_bins: int = 10):
    rows = reliability_table(y, p, n_bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] / n * abs(r["p_mean"] - r["y_mean"]) for r in rows))


def auc(y, p):
    """Rank AUC. DO NOT optimise it: it is dominated by late-session observations
    where the HOD is already final. Reported for context only."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


# ----------------------------------------------------------------------------
# Quantile / distribution metrics
# ----------------------------------------------------------------------------
def pinball(y, q_hat, q):
    y = np.asarray(y, float)
    e = y - np.asarray(q_hat, float)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def crps_from_quantiles(y, Q, taus):
    """CRPS = 2 * integral_0^1 QL_tau dtau (pinball loss identity).

    Q: shape (n, len(taus)). Estimated by averaging over a grid of taus.
    """
    taus = np.asarray(taus, float)
    Q = np.asarray(Q, float)
    return float(2.0 * np.mean([pinball(y, Q[:, i], taus[i]) for i in range(len(taus))]))


def interval_coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


def quantile_score_table(y, Q, taus, Q_ref=None):
    """Pinball per quantile, and gain vs a reference (CARR or half-normal)."""
    out = {}
    for i, t in enumerate(np.asarray(taus, float)):
        row = {"pinball": pinball(y, Q[:, i], t)}
        if Q_ref is not None:
            row["pinball_ref"] = pinball(y, np.asarray(Q_ref)[:, i], t)
            row["skill"] = 1.0 - row["pinball"] / row["pinball_ref"] if row["pinball_ref"] > 0 else np.nan
        out[f"q{t:.2f}"] = row
    return out


# ----------------------------------------------------------------------------
# Economics and significance
# ----------------------------------------------------------------------------
def block_bootstrap_ci(x, block: int = 5, n_boot: int = 5000, seed: int = 0):
    """Block-bootstrap CI: preserves intraday and interday autocorrelation."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < block * 2:
        return {"mean": float(np.mean(x)) if n else np.nan, "ci95": [np.nan, np.nan]}
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return {"mean": float(x.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "p_positive": float(np.mean(means > 0))}


def probabilistic_sharpe(sr, T, skew=0.0, kurt=3.0, sr_star=0.0):
    """PSR (Bailey & Lopez de Prado 2014): P(true SR > sr_star)."""
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(max(T - 1, 1)) / denom))


def deflated_sharpe(sr, T, n_trials, sr_std, skew=0.0, kurt=3.0):
    """DSR: PSR evaluated against the maximum Sharpe expected under H0 after N trials.

    sr_std: standard deviation of the Sharpe ratios observed across the N trials
    (dispersion of the trials). gamma = Euler-Mascheroni constant.
    """
    gamma = 0.5772156649
    N = max(int(n_trials), 2)
    e = np.e
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / N) + gamma * norm.ppf(1 - 1.0 / (N * e)))
    return {"sr0": float(sr0), "dsr": probabilistic_sharpe(sr, T, skew, kurt, sr0)}


def net_pnl(points, contracts, tick_value, tick_size, cost_ticks_rt=1.5):
    """Net P&L in dollars, costs included (commission + slippage in round-trip ticks)."""
    gross = np.asarray(points, float) * contracts * (tick_value / tick_size)
    cost = cost_ticks_rt * tick_value * contracts
    return gross - cost, cost


def r_multiple(entry, exit, stop, direction):
    risk = abs(entry - stop)
    if risk <= 0:
        return np.nan
    return direction * (exit - entry) / risk


# ----------------------------------------------------------------------------
# Overfitting guard rails
# ----------------------------------------------------------------------------
def trial_ledger():
    """MUST be maintained: one row per tested configuration.

    Without this registry the DSR and the PBO cannot be computed and any reported
    performance is indefensible. This is the most frequently omitted step.
    """
    return {"n_trials": 0, "sharpe_by_trial": [], "notes": []}


def leakage_audit(feature_names, forbidden=("hod", "lod", "final", "target", "future",
                                            "close_today", "high_today", "low_today")):
    """Flags feature names that smell of information leakage."""
    bad = [f for f in feature_names if any(k in f.lower() for k in forbidden)]
    return {"suspect_features": bad, "ok": len(bad) == 0}
