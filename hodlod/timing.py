"""
C. Timing : a quelle heure se forment H et L.

Resultat de reference : pour un MB sans derive, l'instant du maximum sur [0,1]
suit la loi de l'arcsinus (troisieme loi de l'arcsinus, Levy 1939/1948) :

    P(T_H <= t) = (2/pi) * arcsin(sqrt(t)),   densite 1/(pi sqrt(t(1-t)))

Donc : U-shape, massif aux deux bouts, et ce n'est PAS une propriete empirique de
la vol intraday, c'est une propriete de tout marche aleatoire continu. Le profil
saisonnier de volatilite rajoute une seconde couche : ce n'est plus t mais F(t),
la fraction de variance ecoulee, qui est l'horloge pertinente.

Ce module fournit : la loi de reference, la transformation en temps de volatilite,
les diagnostics (PIT, KS, log-score) et un modele de hasard en temps discret qui
sert de pont entre le timing et le temps reel.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest, norm


# ----------------------------------------------------------------------------
# C1. Loi de l'arcsinus
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
    """Masse de probabilite de la loi de reference sur des intervalles donnes."""
    bins = np.asarray(bins, float)
    return np.diff(arcsine_cdf(bins))


# ----------------------------------------------------------------------------
# C2. Temps de volatilite
# ----------------------------------------------------------------------------
def clock_to_vol_time(t, F_grid):
    """t en fraction de seance -> F(t) en fraction de variance."""
    grid = np.linspace(0.0, 1.0, len(F_grid))
    return np.interp(np.asarray(t, float), grid, np.asarray(F_grid, float))


def vol_time_to_clock(F, F_grid):
    grid = np.linspace(0.0, 1.0, len(F_grid))
    return np.interp(np.asarray(F, float), np.asarray(F_grid, float), grid)


def arcsine_cdf_voltime(t, F_grid):
    """Loi de reference quand la vol est saisonniere : arcsine en temps de vol."""
    return arcsine_cdf(clock_to_vol_time(t, F_grid))


# ----------------------------------------------------------------------------
# C3. Diagnostics
# ----------------------------------------------------------------------------
def pit_from_argmax(argmax_frac, F_grid):
    """PIT en DEUX etapes : u = ArcsinCDF( F(t_max) ), puis test d'uniformite.

    Pourquoi deux etapes : sous H0, F(t_max) suit une loi ARCSINUS (pas uniforme),
    c'est justement le resultat de Levy. On applique donc ArcsinCDF par-dessus.
    Oublier cette seconde etape est l'erreur la plus frequente dans ce diagnostic.

    Sous H0 (MB + profil de vol connu), u ~ U(0,1).
    PIT en U renverse  => extremes plus souvent au milieu que prevu.
    PIT en cloche     => extremes plus precoces/tardifs que l'arcsinus.
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
        "note": "PIT en cloche => extremes plus precoces/tardifs que l'arcsinus",
    }


def pit_report_clock(argmax_frac, alpha: float = 0.05) -> dict:
    """Meme diagnostic SANS le temps de volatilite (horloge brute).

    Sert de controle negatif : si cette version n'est pas rejetee alors que la
    version en temps de vol l'est, c'est que le profil de vol est mal estime.
    """
    u = arcsine_cdf(np.asarray(argmax_frac, float))
    u = u[np.isfinite(u)]
    ks = kstest(u, "uniform")
    return {"ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
            "reject_uniform": bool(ks.pvalue < alpha)}


def discrete_logscore(counts: np.ndarray, p: np.ndarray) -> float:
    """Log-score d'une distribution discrete (plus petit = mieux)."""
    counts = np.asarray(counts, float)
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-np.sum(counts * np.log(p)) / max(counts.sum(), 1.0))


def brier_multiclass(onehot: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.sum((np.asarray(p) - np.asarray(onehot)) ** 2, axis=1)))


# ----------------------------------------------------------------------------
# C4. Hasard en temps discret (pont vers le temps reel)
# ----------------------------------------------------------------------------
def empirical_hazard(argmax_frac, n_bins: int = 48, F_grid=None):
    """Hasard empirique h_k = P(premier depassement dans le bin k | pas encore depasse).

    Renvoie (bins, h, S) ou S est la survie impliquee : S_k = prod_{j<=k}(1-h_j).
    C'est la version non parametrique de p_final(t) ; elle sert de controle
    au modele de diffusion et de cible pour un classifieur.
    """
    a = np.asarray(argmax_frac, float)
    a = a[np.isfinite(a)]
    edges = np.linspace(0, 1, n_bins + 1)
    if F_grid is not None:
        # on binne en temps de volatilite, plus interpretable
        a = clock_to_vol_time(a, F_grid)
    idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    at_risk = np.cumsum(cnt[::-1])[::-1]
    h = np.where(at_risk > 0, cnt / np.maximum(at_risk, 1e-9), 0.0)
    S = np.cumprod(1.0 - h)
    return edges, h, S


def competing_risk_table(argmax_high, argmin_low, n_bins: int = 24):
    """Table de risques concurrents : H et L se disputent la seance.

    Renvoie, par bin, la frequence de "HOD fixe dans ce bin", "LOD fixe dans ce bin",
    et la probabilite que les DEUX soient deja fixes (etat absorbant du range).
    Utile pour une strategie de fade de range plutot que de retournement d'extreme.
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
    """Instant (fraction de seance) du high et du low, depuis des barres 1 min."""
    h = np.asarray(bars.high.values, float)
    l = np.asarray(bars.low.values, float)
    n = len(h)
    return float(np.argmax(h) / max(n - 1, 1)), float(np.argmin(l) / max(n - 1, 1))
