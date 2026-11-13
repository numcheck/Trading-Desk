"""
Validation : walk-forward purge, metriques propres, couts, sur-ajustement.

Regles dures appliquees ici :
  - aucun look-ahead : le label du jour J (HOD final) n'est disponible qu'a la
    cloture de J, donc tout modele qui predit un evenement de J s'entraine sur <= J-1.
  - purge : on retire du train les jours dont le label chevauche la fenetre de test
    (utile quand la cible est une moyenne glissante ou un horizon a h jours).
  - embargo : on retire quelques jours avant le test pour casser l'autocorrelation
    residuelle (Lopez de Prado 2018, ch. 7).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------
# Decoupage temporel
# ----------------------------------------------------------------------------
def purged_walk_forward(n: int, train: int, test: int, embargo: int = 0,
                        label_horizon: int = 0):
    """Genere (idx_train, idx_test) en avançant dans le temps, sans recouvrement.

    label_horizon : nombre de jours sur lesquels le label du train "voit" le futur.
    Les echantillons de train dont le label deborde sur le test sont retires.
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
# Metriques probabilistes
# ----------------------------------------------------------------------------
def brier(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def brier_skill(y, p, p_ref):
    """BSS = 1 - BS/BS_ref. > 0 : on fait mieux que la reference.

    La reference doit etre le modele analytique (diffusion), pas la climatologie.
    Comparer a la climatologie est trompeur : la base rate de "le HOD courant est
    final" monte vers 1 en fin de seance, donc la climatologie est deja bonne.
    """
    b0 = brier(y, p_ref)
    return float(1.0 - brier(y, p) / b0) if b0 > 0 else np.nan


def logloss(y, p, eps: float = 1e-12):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def reliability_table(y, p, n_bins: int = 10):
    """Table de fiabilite : probabilite predite moyenne vs frequence observee."""
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
    """AUC de rang. A NE PAS optimiser : elle est dominee par les observations de
    fin de seance ou le HOD est deja final. On la rapporte pour le contexte."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


# ----------------------------------------------------------------------------
# Metriques de quantile / distribution
# ----------------------------------------------------------------------------
def pinball(y, q_hat, q):
    y = np.asarray(y, float)
    e = y - np.asarray(q_hat, float)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def crps_from_quantiles(y, Q, taus):
    """CRPS = 2 * integrale_0^1 QL_tau dtau (identite de la perte pinball).

    Q : shape (n, len(taus)). Estimation par moyenne sur une grille de taus.
    """
    taus = np.asarray(taus, float)
    Q = np.asarray(Q, float)
    return float(2.0 * np.mean([pinball(y, Q[:, i], taus[i]) for i in range(len(taus))]))


def interval_coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


def quantile_score_table(y, Q, taus, Q_ref=None):
    """Pinball par quantile, et gain vs une reference (CARR ou demi-normal)."""
    out = {}
    for i, t in enumerate(np.asarray(taus, float)):
        row = {"pinball": pinball(y, Q[:, i], t)}
        if Q_ref is not None:
            row["pinball_ref"] = pinball(y, np.asarray(Q_ref)[:, i], t)
            row["skill"] = 1.0 - row["pinball"] / row["pinball_ref"] if row["pinball_ref"] > 0 else np.nan
        out[f"q{t:.2f}"] = row
    return out


# ----------------------------------------------------------------------------
# Economie et significativite
# ----------------------------------------------------------------------------
def block_bootstrap_ci(x, block: int = 5, n_boot: int = 5000, seed: int = 0):
    """IC bootstrap par blocs : preserve l'autocorrelation intra-jour et inter-jours."""
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
    """PSR (Bailey & Lopez de Prado 2014) : P(SR vrai > sr_star)."""
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2, 1e-12))
    return float(norm.cdf((sr - sr_star) * np.sqrt(max(T - 1, 1)) / denom))


def deflated_sharpe(sr, T, n_trials, sr_std, skew=0.0, kurt=3.0):
    """DSR : PSR evalue contre le Sharpe maximal attendu sous H0 apres N essais.

    sr_std : ecart-type des Sharpe observes sur les N essais (dispersion des trials).
    gamma = constante d'Euler-Mascheroni.
    """
    gamma = 0.5772156649
    N = max(int(n_trials), 2)
    e = np.e
    sr0 = sr_std * ((1 - gamma) * norm.ppf(1 - 1.0 / N) + gamma * norm.ppf(1 - 1.0 / (N * e)))
    return {"sr0": float(sr0), "dsr": probabilistic_sharpe(sr, T, skew, kurt, sr0)}


def net_pnl(points, contracts, tick_value, tick_size, cost_ticks_rt=1.5):
    """P&L net en dollars, couts inclus (commission + slippage en ticks aller-retour)."""
    gross = np.asarray(points, float) * contracts * (tick_value / tick_size)
    cost = cost_ticks_rt * tick_value * contracts
    return gross - cost, cost


def r_multiple(entry, exit, stop, direction):
    risk = abs(entry - stop)
    if risk <= 0:
        return np.nan
    return direction * (exit - entry) / risk


# ----------------------------------------------------------------------------
# Garde-fous anti-sur-ajustement
# ----------------------------------------------------------------------------
def trial_ledger():
    """A tenir OBLIGATOIREMENT : une ligne par configuration testee.

    Sans ce registre, le DSR et le PBO sont incalculables et toute performance
    annoncee est indefendable. C'est le point le plus souvent omis.
    """
    return {"n_trials": 0, "sharpe_by_trial": [], "notes": []}


def leakage_audit(feature_names, forbidden=("hod", "lod", "final", "target", "future",
                                            "close_today", "high_today", "low_today")):
    """Detecte les noms de features qui sentent la fuite d'information."""
    bad = [f for f in feature_names if any(k in f.lower() for k in forbidden)]
    return {"suspect_features": bad, "ok": len(bad) == 0}
