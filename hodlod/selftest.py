"""
Auto-tests Monte-Carlo des formules fermees.

Chaque test verifie une affirmation mathematique du rapport par simulation.
Si un test echoue, c'est la formule (ou son domaine de validite) qui est en cause,
pas la donnee. A lancer : python -m hodlod.selftest
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kstest

from . import amplitude as amp
from . import evaluate as ev
from . import realtime as rt
from . import timing as tm
from . import vol

PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, value, target, tol):
    ok = abs(value - target) <= tol
    _results.append((PASS if ok else FAIL, name, value, target, tol))
    return ok


def mc_paths(n_paths=6000, n_steps=1000, sigma=1.0, profile=None, mu=0.0, seed=0):
    """Marche brownienne a variance saisonniere. Rend les prix cumules (n_paths, n_steps)."""
    rng = np.random.default_rng(seed)
    prof = np.ones(n_steps) / n_steps if profile is None else np.asarray(profile, float)
    prof = prof / prof.sum()
    dt = prof[None, :]
    incr = (mu * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)))
    return np.cumsum(incr, axis=1), prof


# ---------------------------------------------------------------------------
def t_halfnormal():
    """P(H-O <= x) = 2Phi(x/sigma)-1 pour un MB sans derive."""
    p, _ = mc_paths(n_paths=8000, n_steps=1200, sigma=1.0)
    M = p.max(axis=1)
    for x in (0.5, 1.0, 1.6449):
        emp = float(np.mean(M <= x))
        th = 2 * 1.0 * __import__("scipy.stats", fromlist=["norm"]).norm.cdf(x) - 1
        check(f"halfnormal CDF x={x}", emp, th, 0.025)


def t_reflection_nodrift():
    """E[2Phi(d/s)-1] doit egaler la frequence empirique de non-depassement."""
    p, _ = mc_paths(n_paths=5000, n_steps=2000, sigma=1.0)
    half = p.shape[1] // 2
    M = p[:, :half].max(axis=1)
    X = p[:, half - 1]
    d = M - X
    s = np.sqrt(0.5)
    phat = rt.p_no_new_high(d, s)
    emp = float(np.mean(p[:, half:].max(axis=1) <= M))
    check("reflection: mean(p) vs empirical", float(phat.mean()), emp, 0.02)


def t_reflection_drift():
    """Meme test avec derive : la formule de barriere absorbante doit suivre."""
    for mu in (0.4, -0.4):
        p, _ = mc_paths(n_paths=5000, n_steps=2000, sigma=1.0, mu=mu, seed=3)
        half = p.shape[1] // 2
        M = p[:, :half].max(axis=1)
        X = p[:, half - 1]
        d = M - X
        s = np.sqrt(0.5)
        phat = rt.p_no_new_high_drift(d, mu * 0.5, s)
        emp = float(np.mean(p[:, half:].max(axis=1) <= M))
        check(f"drift mu={mu}: mean(p) vs empirical", float(phat.mean()), emp, 0.025)


def t_drift_limit():
    """mu -> 0 doit redonner la formule sans derive."""
    d = np.array([0.3, 1.0, 2.0])
    s = np.array([1.0, 1.0, 1.0])
    a = rt.p_no_new_high_drift(d, np.full(3, 1e-12), s)
    b = rt.p_no_new_high(d, s)
    check("drift limit (max abs err)", float(np.max(np.abs(a - b))), 0.0, 1e-8)


def t_double_barrier():
    """Serie d'images a deux barrieres vs simulation."""
    a = b = 1.0
    s = 1.0
    th = rt.p_range_within(a, b, s)
    # coherence : deux barrieres ne peuvent pas etre plus indulgentes qu'une seule
    assert th < rt.p_no_new_high(a, s), "serie a deux barrieres incoherente"
    p, _ = mc_paths(n_paths=8000, n_steps=2000, sigma=1.0, seed=5)
    emp = float(np.mean((p.min(axis=1) > -a) & (p.max(axis=1) < b)))
    check("double barrier (a=b=1sigma)", th, emp, 0.03)


def t_arcsine_flat():
    """Troisieme loi de l'arcsinus : l'instant du max est arcsine en temps horloge
    quand la vol est plate."""
    p, _ = mc_paths(n_paths=4000, n_steps=1000, sigma=1.0, seed=7)
    am = p.argmax(axis=1) / (p.shape[1] - 1)
    ks = kstest(am, lambda x: tm.arcsine_cdf(x))
    check("arcsine: KS vs loi de reference", float(ks.statistic), 0.0, 0.035)


def t_arcsine_voltime():
    """Avec vol saisonniere : l'horloge brute REJETTE, le temps de vol ne rejette pas."""
    n = 1000
    t = np.linspace(0, 1, n)
    prof = 2.0 - 1.8 * np.sin(np.pi * t)          # U-shape marquee
    F = np.cumsum(prof / prof.sum())
    p, _ = mc_paths(n_paths=4000, n_steps=n, sigma=1.0, profile=prof, seed=11)
    am = p.argmax(axis=1) / (n - 1)
    # horloge brute : PIT sans temps de volatilite
    ks_clock = kstest(tm.arcsine_cdf(am), "uniform").statistic
    # temps de volatilite : PIT a deux etapes
    ks_vol = kstest(tm.pit_from_argmax(am, F), "uniform").statistic
    ok = (ks_clock > 3 * ks_vol) and (ks_vol < 0.04)
    _results.append((PASS if ok else FAIL, "vol time bat l'horloge brute",
                     f"KS_clock={ks_clock:.3f} KS_vol={ks_vol:.3f}", "clock >> vol", 0.0))


def t_carr_recovery():
    """CARR(1,1) : le QMLE doit retrouver les parametres sur donnees simulees."""
    rng = np.random.default_rng(2)
    T = 20000
    om, al, be = 0.25, 0.10, 0.85          # E[R] = om/(1-al-be) = 5
    eps = rng.exponential(1.0, T)
    lam = np.empty(T)
    R = np.empty(T)
    lam[0] = om / (1.0 - al - be)          # depart a la moyenne stationnaire
    R[0] = lam[0] * eps[0]
    for t in range(1, T):                   # meme recursion que le modele
        lam[t] = om + al * R[t - 1] + be * lam[t - 1]
        R[t] = lam[t] * eps[t]
    m = vol.CARR(burn=50).fit(R)
    o, a, b = m.theta_
    check("CARR alpha", float(a), al, 0.06)
    check("CARR beta", float(b), be, 0.05)
    check("CARR persistence", float(a + b), al + be, 0.03)


def t_quantreg_coverage():
    """La regression quantile a tau=0.9 doit couvrir ~90% en test."""
    rng = np.random.default_rng(4)
    n, nt = 2000, 5000
    X = rng.uniform(0, 2, (n + nt, 2))
    y = 1 + 2 * X[:, 0] + (0.5 + 0.5 * X[:, 1]) * rng.standard_normal(n + nt)
    m = amp.fit_quantreg(X[:n], y[:n], q=0.9)
    qh = m.predict(np.column_stack([np.ones(nt), X[n:]]))
    cov = float(np.mean(y[n:] <= qh))
    check("quantreg coverage q=0.9", cov, 0.90, 0.02)


def t_gpd_tail():
    """POT sur une Pareto(alpha=3) : indice de queue 1/3 et quantile q=0.99."""
    rng = np.random.default_rng(6)
    n = 50000
    x = 1.0 / rng.uniform(0, 1, n) ** (1.0 / 3.0)   # Pareto type I, xi = 1/3
    fit = amp.fit_gpd_tail(x, threshold_q=0.90)
    check("GPD xi", fit["xi"], 1.0 / 3.0, 0.10)
    true_q99 = (1 - 0.99) ** (-1.0 / 3.0)
    check("GPD q99 (relatif)", amp.gpd_quantile(0.99, fit) / true_q99, 1.0, 0.15)


def t_brier():
    """Sanity check des scores propres."""
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 20000)
    p_perfect = y.astype(float)
    check("Brier d'un forecast parfait", ev.brier(y, p_perfect), 0.0, 1e-12)
    base = 0.3
    p_clim = np.full_like(y, base, dtype=float)
    p_good = np.clip(base + 0.4 * (y - base), 0, 1)
    check("Brier skill > 0 pour un bon modele", ev.brier_skill(y, p_good, p_clim) > 0, True, 1e-9)


def t_purged_walkforward():
    """Le walk-forward ne doit jamais melanger train et test."""
    ok = True
    for tr, te in ev.purged_walk_forward(100, 40, 10, embargo=2, label_horizon=3):
        if tr.max() >= te.min() or te.min() <= tr.max() + 3:
            ok = False
    check("walk-forward sans recouvrement", ok, True, 1e-9)


def t_deflated_sharpe():
    """Le DSR doit decroitre quand le nombre d'essais augmente."""
    a = ev.deflated_sharpe(1.0, 1000, 2, 0.5)["dsr"]
    b = ev.deflated_sharpe(1.0, 1000, 500, 0.5)["dsr"]
    check("DSR decroit avec n_trials", float(a > b), True, 1e-9)


def t_pinball_ordering():
    """Un quantile mieux estime doit avoir une perte pinball plus faible."""
    rng = np.random.default_rng(9)
    y = rng.standard_normal(5000)
    good = np.full_like(y, 1.2816)
    bad = np.full_like(y, 0.0)
    check("pinball: bon < mauvais", float(amp.pinball(y, good, 0.9) < amp.pinball(y, bad, 0.9)), True, 1e-9)


TESTS = [t_halfnormal, t_reflection_nodrift, t_reflection_drift, t_drift_limit,
         t_double_barrier, t_arcsine_flat, t_arcsine_voltime, t_carr_recovery,
         t_quantreg_coverage, t_gpd_tail, t_brier, t_purged_walkforward,
         t_deflated_sharpe, t_pinball_ordering]


def main():
    for f in TESTS:
        try:
            f()
        except Exception as e:  # noqa: BLE001
            _results.append((FAIL, f.__name__, f"exception: {e}", "-", "-"))
    width = max(len(r[1]) for r in _results)
    n_fail = 0
    print("=" * (width + 42))
    for status, name, val, target, tol in _results:
        n_fail += status == FAIL
        v = f"{val:.5g}" if isinstance(val, (int, float, np.floating)) else str(val)
        t = f"{target:.5g}" if isinstance(target, (int, float, np.floating)) else str(target)
        print(f"{status:4s} | {name:<{width}} | {v:>12s} vs {t:>10s} (tol {tol})")
    print("=" * (width + 42))
    print(f"{len(_results) - n_fail}/{len(_results)} tests passes")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
