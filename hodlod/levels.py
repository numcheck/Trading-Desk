"""
B5. Niveaux de confluence : PDH/PDL, high/low overnight, VWAP, initial balance,
volume profile, chiffres ronds.

Le piege central : "le PDH est touche 70% du temps" ne veut rien dire. Un niveau
proche du prix est touche plus souvent, mecaniquement. Il faut donc :
  1) conditionner sur la distance en sigma,
  2) comparer a un niveau PLACEBO a la meme distance,
  3) definir "rejet" et "cassure" de facon symetrique et sans look-ahead.

Ce module fournit les trois. Les frequences elles-memes doivent etre estimees sur
VOS donnees : je ne les invente pas ici.
"""
from __future__ import annotations

import numpy as np


def reference_levels(prev: dict, overnight: dict, session: dict | None = None) -> dict:
    """Niveaux connus AVANT l'ouverture RTH.

    prev      : {'high','low','close','vwap'} de la seance RTH precedente
    overnight : {'high','low','close','vwap'} de la session ETH (16:00->09:30)
    """
    lv = {
        "pdh": prev.get("high"), "pdl": prev.get("low"), "pdc": prev.get("close"),
        "pd_vwap": prev.get("vwap"),
        "onh": overnight.get("high"), "onl": overnight.get("low"),
        "onc": overnight.get("close"), "on_vwap": overnight.get("vwap"),
    }
    if session:
        lv.update({f"s_{k}": v for k, v in session.items()})
    return {k: float(v) for k, v in lv.items() if v is not None and np.isfinite(v)}


def initial_balance(bars, minutes: int = 30) -> dict:
    """Initial balance : high/low des `minutes` premieres minutes. Connu a 10:00."""
    ib = bars.iloc[:minutes]
    return {"ibh": float(ib.high.max()), "ibl": float(ib.low.min())}


def touch_break(highs, lows, closes, level: float, tol: float,
                side: str = "up") -> dict:
    """Evenement "touche puis rejette" vs "touche puis casse", sans look-ahead.

    side='up'   : on teste un niveau au-dessus (PDH, ONH, IBH).
    Un TOUCH est une meche dans [level-tol, level+tol].
    Une CASSURE est une cloture au-dela de level+tol (ou level-tol si side='down')
    dans les `horizon` barres suivant le premier touch.
    Un REJET est un touch sans cassure et un retour au-dela de tol du cote d'origine.
    """
    h = np.asarray(highs, float)
    l = np.asarray(lows, float)
    c = np.asarray(closes, float)
    n = len(h)
    if side == "up":
        hit = np.where(h >= level - tol)[0]
    else:
        hit = np.where(l <= level + tol)[0]
    if len(hit) == 0:
        return {"touched": 0, "broke": 0, "rejected": 0, "first_hit": np.nan}
    i0 = int(hit[0])
    if side == "up":
        broke = bool(np.any(c[i0:] > level + tol))
        rejected = bool(not broke and np.any(c[i0:] < level - 3 * tol))
    else:
        broke = bool(np.any(c[i0:] < level - tol))
        rejected = bool(not broke and np.any(c[i0:] > level + 3 * tol))
    return {"touched": 1, "broke": int(broke), "rejected": int(rejected),
            "first_hit": i0 / max(n - 1, 1)}


def placebo_levels(open_px: float, sigma_day: float, distances_sigma: np.ndarray,
                   n: int, seed: int = 0, side: str = "up") -> np.ndarray:
    """Niveaux factices a des distances tirees de la distribution reelle.

    C'est le controle indispensable : on compare le taux de rejet d'un VRAI niveau
    au taux de rejet d'un niveau sans signification place a la MEME distance.
    Si l'ecart n'est pas significatif, le "niveau" n'apporte rien.
    """
    rng = np.random.default_rng(seed)
    d = rng.choice(np.asarray(distances_sigma, float), size=n, replace=True)
    d = d * sigma_day
    return open_px + d if side == "up" else open_px - d


def confluence_table(days: list[dict], level_key: str, tol_sigma: float = 0.05,
                     placebo: bool = True, n_boot: int = 2000, seed: int = 0) -> dict:
    """Table de frequences touche/cassure/rejet, avec IC bootstrap et placebo.

    days : liste de dicts, un par seance, contenant au minimum
           {'highs','lows','closes','open','sigma_day','levels'} ou 'levels'
           est le dict produit par reference_levels().
    Retourne les taux reels, les taux placebo et la difference avec IC 95%.
    C'est cette difference qui est le seul chiffre interpretable.
    """
    real, plac, dists = [], [], []
    for d in days:
        lv = d["levels"].get(level_key)
        if lv is None or not np.isfinite(lv):
            continue
        tolv = tol_sigma * d["sigma_day"]
        side = "up" if lv >= d["open"] else "down"
        r = touch_break(d["highs"], d["lows"], d["closes"], lv, tolv, side)
        if not r["touched"]:
            continue
        real.append(r["rejected"])
        dists.append(abs(lv - d["open"]) / max(d["sigma_day"], 1e-9))
        if placebo:
            open_px = float(d["open"])
            sd = float(d["sigma_day"])
            rng = np.random.default_rng(seed + len(real))
            fake = placebo_levels(open_px, sd, np.array(dists), 1, int(rng.integers(1e9)), side)[0]
            rp = touch_break(d["highs"], d["lows"], d["closes"], fake, tolv, side)
            plac.append(rp["rejected"] if rp["touched"] else 0)
    real = np.array(real, float)
    if len(real) == 0:
        return {"n": 0}
    out = {"n": int(len(real)), "reject_rate": float(real.mean()),
           "reject_rate_se": float(real.std(ddof=1) / np.sqrt(len(real))) if len(real) > 1 else np.nan}
    if placebo and plac:
        plac = np.array(plac, float)
        diff = real.mean() - plac.mean()
        rng = np.random.default_rng(seed)
        bs = np.array([rng.choice(real, len(real)).mean() - rng.choice(plac, len(plac)).mean()
                       for _ in range(n_boot)])
        out.update({"placebo_reject_rate": float(plac.mean()),
                    "delta": float(diff),
                    "delta_ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                    "significant": bool(np.quantile(bs, 0.025) > 0 or np.quantile(bs, 0.975) < 0)})
    out["mean_distance_sigma"] = float(np.mean(dists))
    return out


def volume_profile_poc(bars, bin_size: float = 0.25, value_area: float = 0.70) -> dict:
    """Point of Control et value area approximes par histogramme de volume.

    Approximation honnete : sans donnees tick par tick on repartit le volume de
    chaque barre uniformement sur son range. C'est biaise mais utilisable pour un
    niveau de confluence ; ne pas le presenter comme un vrai profil de volume.
    """
    lo = np.floor(bars.low.min() / bin_size) * bin_size
    hi = np.ceil(bars.high.max() / bin_size) * bin_size
    edges = np.arange(lo, hi + bin_size, bin_size)
    if len(edges) < 3:
        return {"poc": float(bars.close.mean()), "val": float(bars.low.min()),
                "vah": float(bars.high.max())}
    vol = np.zeros(len(edges) - 1)
    for _, b in bars.iterrows():
        plo, phi = b.low, b.high
        m = (edges[:-1] < phi) & (edges[1:] > plo)
        k = m.sum()
        if k:
            vol[m] += b.volume / k
    poc_i = int(np.argmax(vol))
    order = np.argsort(vol)[::-1]
    cum, sel = 0.0, []
    for i in order:
        sel.append(i)
        cum += vol[i]
        if cum >= value_area * vol.sum():
            break
    sel = np.sort(sel)
    return {"poc": float(0.5 * (edges[poc_i] + edges[poc_i + 1])),
            "val": float(edges[sel[0]]), "vah": float(edges[sel[-1] + 1])}
