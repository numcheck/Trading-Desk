"""
A/B. Amplitude : distances O->H et O->L attendues.

Convention : on travaille en unites de sigma du jour.
    R+ = (H - O) / sigma_day,   R- = (O - L) / sigma_day,   A = R+ - R-
A est l'asymetrie : c'est elle qui dit si la journee est "up" ou "down", et c'est
la seule des trois qui contient une information directionnelle exploitable.

Benchmark exact (MB sans derive, horizon T, vol sigma) :
    H - O =_d |N(0, sigma^2 T)|   =>   P(R+ <= x) = 2 Phi(x) - 1
car P(max < a) = 2Phi(a/s)-1. D'ou E[R+] = sqrt(2/pi) = 0.7979 et
q_90(R+) = Phi^{-1}(0.95) = 1.6449. Tout ecart systematique a cette loi est une
prime de queues/jumps/derive a mesurer, pas a supposer.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import genpareto, norm


# ----------------------------------------------------------------------------
# A5. Benchmark demi-normal
# ----------------------------------------------------------------------------
def halfnormal_cdf(x, sigma_rem=1.0):
    x = np.asarray(x, float)
    return 2.0 * norm.cdf(x / max(sigma_rem, 1e-12)) - 1.0


def halfnormal_quantile(q, sigma_rem=1.0):
    """q_alpha de (H-O) sous MB sans derive : sigma * Phi^{-1}((1+alpha)/2)."""
    q = np.asarray(q, float)
    return sigma_rem * norm.ppf(0.5 + 0.5 * q)


def amplitude_multiplier(x, sigma_rem=1.0, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95)):
    """q_empirique / q_theorique de R+ pour chaque quantile.

    > 1 partout => queues plus epaisses que le MB (sauts, vol stochastique, derive).
    Le profil du ratio selon alpha est le diagnostic le plus informatif :
      - ratio plat > 1  : simple mise a l'echelle (vol mal estimee),
      - ratio croissant : queues epaisses (jumps),
      - ratio > 1 aux deux bouts : bornes de monitoring + sauts.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    emp = np.quantile(x, qs)
    th = halfnormal_quantile(np.array(qs), sigma_rem)
    return dict(zip([f"q{int(q*100)}" for q in qs], (emp / th).tolist()))


# ----------------------------------------------------------------------------
# A6. EVT : queues de l'amplitude normalisee
# ----------------------------------------------------------------------------
def fit_gpd_tail(x, threshold_q: float = 0.90) -> dict:
    """Ajustement GPD par exces au-dela du quantile threshold_q (Peaks-Over-Threshold).

    Modele : pour u = q_{F}(x) et Y = x - u | x > u,  Y ~ GPD(xi, sigma).
    Sortie : dictionnaire avec u, xi (forme = indice de queue), sigma (echelle),
    et les effectifs. xi > 0 => queue polynomiale, xi = 0 => exponentielle.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    u = float(np.quantile(x, threshold_q))
    exc = x[x > u] - u
    if len(exc) < 50:
        raise ValueError("trop peu d'exces pour ajuster une GPD (<50)")
    xi, loc, scale = genpareto.fit(exc, floc=0.0)
    return {"u": u, "xi": float(xi), "sigma": float(scale),
            "n": int(len(x)), "n_excess": int(len(exc)),
            "threshold_q": float(threshold_q)}


def gpd_quantile(q, fit: dict) -> float:
    """Quantile POT au-dela du seuil : u + (sigma/xi)[((1-q)/(1-F_u))^{-xi} - 1]."""
    u, xi, sig = fit["u"], fit["xi"], fit["sigma"]
    Fu = fit["threshold_q"]
    q = float(q)
    if q <= Fu:
        raise ValueError("q doit depasser le quantile de seuil")
    if abs(xi) < 1e-8:
        return float(u - sig * np.log((1.0 - q) / (1.0 - Fu)))
    return float(u + (sig / xi) * (((1.0 - q) / (1.0 - Fu)) ** (-xi) - 1.0))


def gpd_tail_prob(x, fit: dict) -> float:
    """P(X > x) selon la GPD ajustee (x > u)."""
    u, xi, sig = fit["u"], fit["xi"], fit["sigma"]
    Fu = fit["threshold_q"]
    z = max(x - u, 0.0)
    if abs(xi) < 1e-8:
        return float((1 - Fu) * np.exp(-z / sig))
    return float((1 - Fu) * max(1.0 + xi * z / sig, 1e-12) ** (-1.0 / xi))


# ----------------------------------------------------------------------------
# A7. Regression quantile et boosting pinball
# ----------------------------------------------------------------------------
def pinball(y, q_hat, q):
    """Perte pinball (asymetrique absolue) : la seule perte propre pour un quantile."""
    y = np.asarray(y, float)
    e = y - np.asarray(q_hat, float)
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


def fit_quantreg(X, y, q: float = 0.5):
    """Regression quantile lineaire (Koenker & Bassett 1978) via statsmodels."""
    import statsmodels.api as sm
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return sm.QuantReg(np.asarray(y, float), sm.add_constant(X)).fit(q=q)


class QuantileBooster:
    """Gradient boosting avec perte pinball (sklearn, loss='quantile').

    Ce qu'il estime : q_alpha(R+ | X), le quantile conditionnel de l'amplitude,
    pas la moyenne. Avantage sur la regression lineaire : interactions
    (regime de vol x jour d'annonce x gap), non-linearites, robustesse aux outliers.
    Risque : sur-ajustement rapide avec peu de jours ; garder max_depth <= 3 et
    early stopping sur un fold de validation temporel.
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
    """Un jeu de quantiles doit etre monotone en alpha. Compte les croisements.

    q_preds : {alpha: predictions}. Un taux de croisement non nul signale un
    modele instable (frequent avec les forêts/boosting par quantile independants).
    """
    alphas = sorted(q_preds)
    P = np.column_stack([q_preds[a] for a in alphas])
    return np.mean(np.diff(P, axis=1) < 0, axis=1)


# ----------------------------------------------------------------------------
# B4. Classifieur d'asymetrie : "quel cote s'etend ?"
# ----------------------------------------------------------------------------
def asym_target(R_plus, R_minus):
    """1 si la branche haute domine, 0 sinon. Sous MB sans derive : P = 1/2."""
    return (np.asarray(R_plus, float) > np.asarray(R_minus, float)).astype(int)


def fit_asym_classifier(X, y):
    """Classifieur calibre (gradient boosting + calibration isotonique)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    base = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                          learning_rate=0.05, l2_regularization=1.0,
                                          random_state=0)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(np.asarray(X, float), np.asarray(y, int))
    return clf


def amplitude_features(df) -> np.ndarray:
    """Features pre-ouverture pour R+/R- (aucune donnee du jour J).

    df doit contenir, pour chaque jour : rv_prev (1,5,22), var_pk_prev, range_prev_sigma,
    gap, overnight_range, vix, vix_chg, dow, is_event, close_pos_in_range_prev.
    """
    cols = ["log_rv1", "log_rv5", "log_rv22", "log_pk_prev", "range_prev_sigma",
            "gap_sigma", "on_range_sigma", "vix", "vix_chg", "dow", "is_event",
            "close_pos_prev"]
    return df[[c for c in cols if c in df.columns]].to_numpy(float)
