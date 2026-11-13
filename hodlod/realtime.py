"""
D. Temps reel : P(le HOD/LOD courant est le final | information a t).

Le noyau est une probabilite de survie de maximum. On donne :
  - la formule sans derive (principe de reflexion),
  - son extension avec derive,
  - le cas a deux barrieres (range),
  - le temps de volatilite,
  - le constructeur de features et la couche de calibration.

Le point crucial : ces formules supposent un MB a volatilite CONSTANTE. Le vrai
marche viole 3 hypotheses (derive, queues epaisses/sauts, vol stochastique). La
couche de calibration empirique est ce qui rend l'objet utilisable ; la formule
sert de benchmark que le ML doit battre, pas de verite.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


# ----------------------------------------------------------------------------
# D1. Formules fermees
# ----------------------------------------------------------------------------
def p_no_new_high(d, s):
    """P(le maximum courant n'est pas depasse sur l'horizon restant), sans derive.

    d : distance (en prix) entre le max courant et le prix courant, d >= 0
    s : ecart-type du prix sur l'horizon restant = sigma * sqrt(tau)

    Derivation : soit M_t le max courant et X_t le prix. Pour un MB sans derive
    partant de X_t, P(max_{[t,T]} < M_t) = P(max_{u<=tau} (X_t + sigma W_u) < M_t)
    = 2*Phi(d/s) - 1, avec d = M_t - X_t. (Principe de reflexion :
    P(M_tau >= a) = 2 P(B_tau >= a) => P(M_tau < a) = 1 - 2(1-Phi(a/s)) = 2Phi(a/s)-1.)

    C'est exactement la formule de l'enonce. Limites : d/s -> 0 => p -> 0
    (au plus haut avec le prix au plus haut, un nouveau plus haut est quasi certain),
    d/s grand => p -> 1 (mais lentement).
    """
    d = np.asarray(d, float)
    s = np.maximum(np.asarray(s, float), 1e-12)
    return np.clip(2.0 * norm.cdf(d / s) - 1.0, 0.0, 1.0)


def p_no_new_high_drift(d, mu, s):
    """Meme probabilite avec derive.

    d  : distance au maximum courant (prix)
    mu : derive sur l'horizon restant (prix) = mu_rate * tau
    s  : sigma * sqrt(tau)

    Formule de la barriere absorbante :
      P(max < a) = Phi((a - mu)/s) - exp(2*mu*a/s^2) * Phi((-a - mu)/s)
    Verifie : mu -> 0 redonne 2Phi(a/s) - 1.
    """
    from scipy.special import log_ndtr
    d = np.atleast_1d(np.asarray(d, float))
    s = np.maximum(np.atleast_1d(np.asarray(s, float)), 1e-12)
    mu = np.broadcast_to(np.asarray(mu, float), d.shape).astype(float)
    # mu nul -> formule sans derive (evite toute annulation numerique)
    zero = np.abs(mu) < 1e-12 * s
    mu_safe = np.where(zero, 0.0, mu)
    # exp(2 mu d / s^2) * Phi((-d-mu)/s) calcule en log pour rester stable
    log_term = log_ndtr((-d - mu_safe) / s) + 2.0 * mu_safe * d / s ** 2
    general = norm.cdf((d - mu_safe) / s) - np.exp(np.clip(log_term, -700.0, 700.0))
    out = np.where(zero, 2.0 * norm.cdf(d / s) - 1.0, general)
    return np.clip(out, 0.0, 1.0)


def p_range_within(a, b, s, n_terms: int = 200):
    """P(-a < min et max < b) pour un MB sans derive, ecart-type total s, depart 0.

    Methode des images a DEUX barrieres absorbantes :
      sum_k [ Phi((b + 2kL)/s) - Phi((-a + 2kL)/s) ]
    - sum_k [ Phi((b + 2a + 2kL)/s) - Phi((a + 2kL)/s) ],   L = a + b.
    Le second somme est indispensable : l'oublier surestime massivement la
    probabilite (0.68 au lieu de 0.37 pour a=b=s). Verifie en Monte-Carlo.

    Limite b -> inf : la seconde somme -> 1 - Phi(a/s) et le total -> 2Phi(a/s)-1,
    ce qui redonne bien la formule a une barriere.
    Convergence rapide si min(a,b)/s >= 0.25 ; lente en dessous (garder n_terms eleve
    ou passer en Monte-Carlo).
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
    """sigma sur l'horizon restant, en temps de volatilite.

    F_now = fraction de variance deja ecoulee (0 a l'ouverture, 1 a la cloture).
    sigma_remaining = sigma_day * sqrt(1 - F_now).
    C'est LE point ou l'on gagne le plus vs temps horloge : la derniere demi-heure
    porte typiquement bien plus de variance que la moyenne uniforme.
    """
    return float(sigma_day * np.sqrt(max(1.0 - F_now, 0.0)))


def p_final_now(price, running_max, running_min, sigma_day, F_now,
                side: str = "high", drift_bps: float = 0.0) -> float:
    """Wrapper pratique : probabilite que le HOD (ou LOD) courant soit le final."""
    s = remaining_sigma(sigma_day, F_now)
    if s <= 0:
        return 1.0
    if side == "high":
        d = max(running_max - price, 0.0)
        return float(p_no_new_high_d(d, s)) if drift_bps == 0 else \
            float(p_no_new_high_drift(np.array([d]), drift_bps * price * 1e-4, np.array([s]))[0])
    d = max(price - running_min, 0.0)
    if drift_bps == 0:
        return float(p_no_new_high(d, s))
    return float(p_no_new_high_drift(np.array([d]), -drift_bps * price * 1e-4, np.array([s]))[0])


# ----------------------------------------------------------------------------
# D2. Version moyennee sur la volatilite stochastique
# ----------------------------------------------------------------------------
def p_final_mixture(d, sigma_day, F_now, vol_of_vol: float = 0.35,
                    n_nodes: int = 9) -> float:
    """Integre la formule sur une distribution log-normale de sigma (vol stochastique).

    Effet : melanger des sigma fait apparaitre des queues plus epaisses, ce qui
    AUGMENTE P(nouveau plus haut) loin de l'extreme -> p_final baisse.
    vol_of_vol = ecart-type de log(sigma) (0.3-0.5 typique sur indices).
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
# D3. Features temps reel (aucun look-ahead)
# ----------------------------------------------------------------------------
def session_features(bars, i: int, sigma_day: float, F_grid: np.ndarray,
                     ctx: dict | None = None) -> dict:
    """Features disponibles a la barre i (inclusivement), pour une seance.

    bars : DataFrame indexe par minute avec open/high/low/close/volume.
    F_grid : fraction de variance cumulee, meme longueur que bars.
    ctx : niveaux de reference calculees AVANT l'ouverture (pdh, pdl, onh, onl, vwap...).

    Regle d'or : toute variable utilisee ici doit etre connue a la fin de la barre i.
    Le HOD final n'entre JAMAIS dans les features.
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
    """Calibration isotonique avec repli lineaire hors bornes.

    Pourquoi calibrer : le modele de diffusion est sur-confiant dans les regimes
    calmes et sous-confiant apres un choc. La calibration doit etre estimee sur un
    echantillon distinct (walk-forward), jamais sur le train du classifieur.
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
        """Pente de calibration : 1 = parfait, <1 = sur-confiance, >1 = sous-confiance."""
        p = np.asarray(p, float)
        y = np.asarray(y, float)
        if p.std() == 0:
            return np.nan
        return float(np.polyfit(p, y, 1)[0])
