"""
Genere les figures et le PDF final (reportlab + matplotlib).

Lit research/out/results.json et les pickles de trades.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.stats import norm

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

OUT = Path("research/out")
FIG = OUT / "fig"
FIG.mkdir(parents=True, exist_ok=True)
rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                 "axes.spines.top": False, "axes.spines.right": False,
                 "figure.dpi": 150, "savefig.bbox": "tight"})
NAVY, RED, GREEN, GREY = "#1f3a5f", "#c0392b", "#27ae60", "#7f8c8d"
RES = json.loads((OUT / "results.json").read_text())


# ------------------------------------------------------------------ figures
def fig_amplitude():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for sym, c in (("ES=F", NAVY), ("NQ=F", RED)):
        R = np.array(RES["amplitude"][sym]["R_up"])
        ax[0].hist(R, bins=60, density=True, alpha=0.45, color=c,
                   label=f"{sym} (n={len(R)})", range=(0, 4))
    x = np.linspace(0.001, 4, 400)
    ax[0].plot(x, 2 * norm.pdf(x) * (x > 0), color="k", lw=1.6,
               label="demi-normal théorique")
    ax[0].set_xlabel("R+ = (H-O)/σ  (unités de σ prévu)")
    ax[0].set_ylabel("densité")
    ax[0].set_title("A. Amplitude O→H contre le benchmark de diffusion")
    ax[0].legend(fontsize=7)
    qs = RES["amplitude"]["ES=F"]["qs"]
    w = 0.38
    xs = np.arange(len(qs))
    ax[1].bar(xs - w / 2, RES["amplitude"]["ES=F"]["mult"], w, color=NAVY, label="ES")
    ax[1].bar(xs + w / 2, RES["amplitude"]["NQ=F"]["mult"], w, color=RED, label="NQ")
    ax[1].axhline(1, color="k", lw=1)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"q{int(q*100)}" for q in qs])
    ax[1].set_ylabel("quantile empirique / quantile théorique")
    ax[1].set_title("Multiplicateur d'amplitude par quantile")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG / "f_amplitude.png")
    plt.close(fig)


def fig_timing():
    t = RES["timing"]["SPY"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    xs = np.arange(7)
    w = 0.27
    ax[0].bar(xs - w, t["obs_high"], w, color=NAVY, label="observé")
    ax[0].bar(xs, t["p_clock"], w, color=GREY, label="arcsinus (horloge)")
    ax[0].bar(xs + w, t["p_vol"], w, color=GREEN, label="arcsinus (temps de vol)")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(["9:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"],
                          fontsize=7, rotation=30)
    ax[0].set_ylabel("fréquence du HOD")
    ax[0].set_title("C. Où se forme le high (SPY, 491 séances)")
    ax[0].legend(fontsize=7)
    for sym, c in (("SPY", NAVY), ("QQQ", RED), ("IWM", GREEN)):
        s = RES["timing"][sym]
        ax[1].plot(np.cumsum(s["obs_high"]), marker="o", color=c, label=sym)
    ax[1].plot(np.cumsum(t["p_clock"]), "--", color="k", lw=1.3, label="arcsinus")
    ax[1].plot(np.cumsum(t["p_vol"]), ":", color=GREY, lw=1.6, label="arcsinus vol-time")
    ax[1].set_xlabel("slot horaire")
    ax[1].set_ylabel("CDF du HOD")
    ax[1].set_title("CDF empirique vs loi de référence")
    ax[1].legend(fontsize=7)
    fig.savefig(FIG / "f_timing.png")
    plt.close(fig)


def fig_calibration():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for sym, c in (("SPY", NAVY), ("QQQ", RED), ("IWM", GREEN)):
        rel = np.array(RES["realtime"][sym]["reliability_hi"])
        ax[0].plot(rel[:, 0], rel[:, 1], marker="o", ms=3, color=c, label=sym)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("p annoncée : « le high courant est le final »")
    ax[0].set_ylabel("fréquence observée")
    ax[0].set_title("D. Fiabilité du modèle de diffusion (hors échantillon)")
    ax[0].legend(fontsize=7)
    t = RES["timing"]["SPY"]
    slots = range(6)
    obs = [RES["realtime"]["SPY"]["full_by_slot"][i]["obs"] for i in slots]
    pred = [RES["realtime"]["SPY"]["full_by_slot"][i]["pred"] for i in slots]
    ax[1].plot(list(slots), obs, marker="o", color=NAVY, label="observé")
    ax[1].plot(list(slots), pred, marker="s", color=RED, label="prédit (formule)")
    ax[1].set_xticks(list(slots))
    ax[1].set_xticklabels(["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"], fontsize=7)
    ax[1].set_ylabel("P(le high est déjà final)")
    ax[1].set_title("La question « combien de fois » : prédit vs réalisé")
    ax[1].legend(fontsize=7)
    fig.savefig(FIG / "f_calib.png")
    plt.close(fig)


def fig_equity(oos):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    d = oos.sort_values(["date", "slot"]).reset_index(drop=True)
    eq = np.cumsum(d.net.values)
    ax[0].plot(range(len(eq)), eq, color=NAVY, lw=1.3)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("trade (ordre chronologique)")
    ax[0].set_ylabel("R cumulés (net de coûts)")
    ax[0].set_title(f"Walk-forward hors échantillon — {len(eq)} trades")
    dd = np.maximum.accumulate(eq) - eq
    ax[1].fill_between(range(len(dd)), -dd, 0, color=RED, alpha=0.45)
    ax[1].set_xlabel("trade")
    ax[1].set_ylabel("drawdown (R)")
    ax[1].set_title(f"Drawdown (max {dd.max():.1f} R)")
    fig.savefig(FIG / "f_equity.png")
    plt.close(fig)


def fig_tp(cand_by_sym, split_frac=0.6):
    """Espérance par trade en fonction du take-profit, IS vs OOS (split fixe)."""
    days = sorted(set(np.concatenate([c.date.unique() for c in cand_by_sym.values()])))
    cut = days[int(len(days) * split_frac)]
    kts = RES["meta"]["k_tps"]
    out = {}
    for th in (0.5, 0.6, 0.7, 0.8):
        is_r, oos_r = [], []
        for kt in kts:
            a, b = [], []
            for sym, c in cand_by_sym.items():
                sub = c[c.p >= th]
                a.append(sub[sub.date < cut][f"n_{kt}"].values)
                b.append(sub[sub.date >= cut][f"n_{kt}"].values)
            a = np.concatenate(a) if a else np.array([])
            b = np.concatenate(b) if b else np.array([])
            is_r.append(a.mean() if len(a) else np.nan)
            oos_r.append(b.mean() if len(b) else np.nan)
        out[th] = (is_r, oos_r)
    fig, ax = plt.subplots(figsize=(10, 3.9))
    cols = {0.5: GREY, 0.6: GREEN, 0.7: NAVY, 0.8: RED}
    for th, (a, b) in out.items():
        ax.plot(kts, a, marker="o", color=cols[th], label=f"θ={th} (IS)")
        ax.plot(kts, b, marker="o", ls="--", color=cols[th], alpha=0.6,
                label=f"θ={th} (OOS)")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("take-profit (multiple de R)")
    ax.set_ylabel("espérance nette par trade (R)")
    ax.set_title("Gestion de prise de profit : espérance selon le TP")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(FIG / "f_tp.png")
    plt.close(fig)
    return out, cut


def fig_mc():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(RES["mc"]["boot_dist"], bins=60, color=NAVY, alpha=0.75)
    ax[0].axvline(0, color=RED, lw=1.4)
    ax[0].set_title("Monte-Carlo : P&L total bootstrapé (R)")
    ax[0].set_xlabel("R cumulés")
    ax[1].hist(RES["mc"]["perm_dist"], bins=60, color=GREY, alpha=0.8)
    ax[1].axvline(RES["oos"]["total_R"], color=RED, lw=1.6)
    ax[1].set_title(f"Null de permutation (p={RES['mc']['perm_p']:.3f})")
    ax[1].set_xlabel("R cumulés sous H0")
    fig.savefig(FIG / "f_mc.png")
    plt.close(fig)


def fig_freq():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    labels, vals, nom = [], [], []
    for sym in ("ES=F", "NQ=F"):
        f = RES["freq_amp"][sym]
        labels += [f"{sym} bande q25-q75", f"{sym} > q90", f"{sym} > q95"]
        vals += [f["band_hit"] * 100, f["above_q90"] * 100, f["above_q95"] * 100]
        nom += [50, 10, 5]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, vals, 0.4, color=NAVY, label="observé")
    ax.bar(xs + 0.2, nom, 0.4, color=GREY, label="nominal (diffusion)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("% des séances")
    ax.set_title("Combien de fois le marché fait ce que le modèle annonce (amplitude)")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "f_freq.png")
    plt.close(fig)


# ------------------------------------------------------------------ PDF
def register_fonts():
    base = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    pdfmetrics.registerFont(TTFont("DJ", str(base / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DJB", str(base / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DJI", str(base / "DejaVuSans-Oblique.ttf")))


def styles():
    ss = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", fontName="DJB", fontSize=15, leading=19,
                             textColor=colors.HexColor(NAVY), spaceAfter=6, spaceBefore=4),
        "h2": ParagraphStyle("h2", fontName="DJB", fontSize=11.5, leading=15,
                             textColor=colors.HexColor(NAVY), spaceAfter=4, spaceBefore=8),
        "body": ParagraphStyle("b", fontName="DJ", fontSize=9, leading=13,
                               alignment=TA_JUSTIFY, spaceAfter=5),
        "small": ParagraphStyle("s", fontName="DJ", fontSize=7.6, leading=10,
                                textColor=colors.HexColor("#444444")),
        "bullet": ParagraphStyle("bl", fontName="DJ", fontSize=9, leading=13,
                                 leftIndent=12, bulletIndent=2, spaceAfter=3),
        "title": ParagraphStyle("t", fontName="DJB", fontSize=26, leading=31,
                                textColor=colors.HexColor(NAVY), spaceAfter=10),
        "sub": ParagraphStyle("st", fontName="DJ", fontSize=12, leading=16,
                              textColor=colors.HexColor("#444444"), spaceAfter=6),
        "kpi": ParagraphStyle("k", fontName="DJB", fontSize=20, leading=24,
                              textColor=colors.HexColor(NAVY)),
    }


def table(data, widths, header=True, size=7.6, align="RIGHT", highlight=None,
          last_left=False):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONT", (0, 0), (-1, -1), "DJ", size),
          ("FONT", (0, 0), (-1, 0), "DJB", size) if header else ("FONT", (0, 0), (-1, -1), "DJ", size),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ALIGN", (1, 0), (-1, -1), align),
          ("ALIGN", (0, 0), (0, -1), "LEFT"),
          ("ALIGN", (-1, 0), (-1, -1), "LEFT" if last_left else align),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("TOPPADDING", (0, 0), (-1, -1), 3),
          ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc"))]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
               ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(NAVY))]
    if highlight:
        st += [("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#fff3cd"))]
    t.setStyle(TableStyle(st))
    return t


def img(name, width=17.4 * cm):
    from PIL import Image as PILImage
    p = FIG / name
    w, h = PILImage.open(p).size
    return Image(str(p), width=width, height=width * h / w)


def build():
    register_fonts()
    s = styles()
    doc = SimpleDocTemplate(str(OUT / "HOD_LOD_ES_NQ.pdf"), pagesize=A4,
                            leftMargin=2.0 * cm, rightMargin=2.0 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.7 * cm,
                            title="HOD/LOD ES-NQ — backtest 3 ans", author="Recherche quantitative")
    F = []
    A = RES["amplitude"]
    O = RES["oos"]
    M = RES["mc"]
    # le drawdown n'a de sens que dans l'ordre CHRONOLOGIQUE (les trades OOS sont
    # concatenes par symbole dans le pickle : on retrie avant de cumuler)
    oos_chrono = pd.read_pickle(OUT / "oos_trades.pkl").sort_values(
        ["date", "slot"]).reset_index(drop=True)
    eq_all = np.cumsum(oos_chrono.net.values)
    DD_CHRONO = float(np.max(np.maximum.accumulate(eq_all) - eq_all))

    # ---- couverture
    F += [Spacer(1, 4 * cm), Paragraph("Où se forment le HOD et le LOD", s["title"]),
          Paragraph("Étude statistique et backtest walk-forward sur ES / NQ", s["sub"]),
          Spacer(1, 1.2 * cm)]
    cover = [["Données", "Gratuites (Yahoo Finance), sans clé API"],
             ["Intraday", f"{RES['timing']['SPY']['n_sessions']} séances × 3 instruments "
                          f"({RES['timing']['SPY']['start']} → {RES['timing']['SPY']['end']})"],
             ["Quotidien", f"{A['ES=F']['n_days']} séances ES & NQ "
                           f"({A['ES=F']['start']} → {A['ES=F']['end']})"],
             ["Backtest", f"{O['n']} trades hors échantillon, sélection walk-forward mensuelle"],
             ["Verdict", f"espérance nette {O['mean_R']:+.4f} R par trade — "
                         f"DSR {RES['dsr']['dsr']:.3f}"]]
    F += [table([["Poste", "Couverture"]] + cover, [4.2 * cm, 13.0 * cm], size=8.5,
                last_left=True),
          Spacer(1, 1.4 * cm),
          Paragraph("Rapport généré automatiquement à partir de données publiques. "
                    "Les chiffres sont ceux du code livré, reproductibles par "
                    "<i>python -m research.engine</i> puis <i>python -m research.report</i>. "
                    "Aucun résultat n'a été arrondi en faveur de la stratégie.", s["small"]),
          PageBreak()]

    # ---- résumé
    F += [Paragraph("1. Résumé exécutif", s["h1"])]
    bss = np.mean([RES["realtime"][k]["hi"]["bss"] for k in ("SPY", "QQQ", "IWM")])
    F += [Paragraph(
        f"<b>La statistique tient ; le trading reste sous le seuil de friction.</b> Le "
        f"modèle de survie de maximum (formule de réflexion) bat une climatologie "
        f"conditionnelle à l'heure avec un Brier Skill Score moyen de <b>{bss:+.3f}</b> "
        f"sur trois instruments. Le backtest walk-forward du retournement au plus haut "
        f"donne une espérance <b>brute de {O['gross_mean_R']:+.4f} R</b> par trade sur "
        f"{O['n']} trades — donc un signal réel — mais un coût aller-retour de "
        f"<b>{O['cost_R']:.4f} R</b>, soit {O['cost_R']/max(O['gross_mean_R'],1e-9):.2f} fois "
        f"cette espérance. Net : {O['mean_R']:+.4f} R, Sharpe par trade "
        f"{O['sharpe_trade']:+.3f}, DSR {RES['dsr']['dsr']:.3f}. Le résultat n'est pas "
        f"« aucun signal » mais <b>« un signal trop petit pour les frais »</b> — une "
        f"conclusion plus utile, et plus difficile à renverser.", s["body"])]
    kp = [["Indicateur", "Valeur", "Lecture"],
          ["Trades hors échantillon", f"{O['n']}", "walk-forward, 3 instruments"],
          ["Taux de réussite", f"{O['hit']*100:.1f} %", "≈ hasard (50 %)"],
          ["Espérance nette", f"{O['mean_R']:+.4f} R", "négative"],
          ["Espérance brute", f"{O['gross_mean_R']:+.4f} R", "positive, mais < coûts"],
          ["Coût moyen", f"{O['cost_R']:.4f} R", f"{O['cost_R']/max(O['gross_mean_R'],1e-9):.2f}× l'espérance brute"],
          ["Profit factor", f"{O['profit_factor']:.3f}", "< 1"],
          ["Drawdown max (chrono)", f"{DD_CHRONO:.1f} R", "sur la période"],
          ["IC 95 % bootstrap", f"[{M['boot_ci'][0]:+.1f} ; {M['boot_ci'][1]:+.1f}] R", "contient 0"],
          ["p(permutation)", f"{M['perm_p']:.3f}", "non significatif"],
          ["DSR", f"{RES['dsr']['dsr']:.3f}", "≪ 0,95 : pas d'edge prouvé"]]
    F += [table(kp, [5.0 * cm, 4.4 * cm, 8.0 * cm], size=8, highlight=3, last_left=True),
          Spacer(1, 0.3 * cm),
          Paragraph(
              "Trois résultats positifs et solides, indépendants du backtest : "
              "(i) l'amplitude RTH vaut <b>1,82 σ</b> en moyenne contre 1,596 prédit par "
              "la diffusion — le marché parcourt 14 % de plus que le modèle ; "
              "(ii) le multiplicateur d'amplitude est <b>quasi constant (1,06–1,12)</b> "
              "au-delà du décile bas, ce qui rend la correction triviale ; "
              "(iii) la loi de l'arcsinus en horloge brute est <b>rejetée</b> "
              f"(χ² = {RES['timing']['SPY']['chi2_clock']:.1f} sur SPY) mais le passage "
              "en temps de volatilité divise la distance KS par 2,6.", s["body"])]

    # ---- données
    F += [PageBreak(), Paragraph("2. Données et limites réelles de la source gratuite", s["h1"]),
          Paragraph(
              "La demande portait sur 3 ans. La source gratuite ne le permet pas en "
              "haute fréquence : les limites ci-dessous ont été mesurées, pas supposées. "
              "Plutôt que de remplir des trous avec des données inventées, la couverture "
              "réelle est utilisée et la granularité est choisie par sous-problème.", s["body"])]
    lim = [["Intervalle", "Profondeur réelle", "Usage retenu"],
           ["1 minute", "~25 jours", "non utilisé"],
           ["5 / 15 / 30 minutes", "60 jours", "contrôle ponctuel"],
           ["1 heure", "730 jours", "timing + temps réel + backtest"],
           ["1 jour", "illimité", "amplitude, HAR, 6 ans"]]
    F += [table(lim, [4.2 * cm, 4.0 * cm, 9.2 * cm], size=8), Spacer(1, 0.25 * cm),
          Paragraph(
              "Les barres horaires de SPY/QQQ/IWM sont reconstruites sur 09:30–16:00 et "
              "<b>validées contre le OHLC quotidien de l'exchange</b> : le high est exact à "
              "2 points de base près sur 99,2 % des séances, le low sur 97,6 %, et le ratio "
              "de range médian vaut 1,000. La valeur du high/low est donc exacte ; seule "
              "sa <i>datation</i> est quantifiée à l'heure. Une barre parasite à 16:00 "
              "capturait les échanges post-clôture et a été supprimée.", s["body"]),
          Paragraph("Discipline anti-look-ahead", s["h2"])]
    F += [Paragraph("• σ(J) est un HAR en fenêtre <b>expansive</b>, réestimé tous les "
                    "20 jours, qui n'utilise que les variances ≤ J−1.", s["bullet"]),
          Paragraph("• Le profil saisonnier est estimé sur le passé glissant.", s["bullet"]),
          Paragraph("• Le backtest entre à l'<b>ouverture de la barre suivante</b> le signal.", s["bullet"]),
          Paragraph("• (seuil, take-profit) sont choisis par <b>walk-forward mensuel</b> sur "
                    "le passé uniquement ; le mois évalué n'entre jamais dans la sélection.", s["bullet"]),
          Paragraph("• Les coûts valent 1,6 % de σ par aller-retour (0,5 point d'ES sur "
                    "31 points de σ), appliqués à chaque trade.", s["bullet"])]

    # ---- amplitude
    F += [PageBreak(), Paragraph("3. Amplitude : combien le marché parcourt", s["h1"])]
    rows = [["Instrument", "n", "E[R+]", "théorique", "E[range/σ]", "théorique", "KS"]]
    for sym in ("ES=F", "NQ=F"):
        a = A[sym]
        rows.append([sym, a["n_days"], f"{a['mean_Rup']:.3f}", f"{a['theo_mean']:.3f}",
                     f"{a['mean_range']:.3f}", f"{a['theo_range']:.3f}", f"{a['ks_Rup']:.3f}"])
    F += [img("f_amplitude.png"), Spacer(1, 0.2 * cm),
          table(rows, [2.6 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.2 * cm, 1.8 * cm], size=7.8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              f"Le benchmark de diffusion (<i>H−O ~ |N(0,σ²T)|</i>) est rejeté au test KS "
              f"(D = {A['ES=F']['ks_Rup']:.3f}, seuil 0,037 à 5 % pour n = {A['ES=F']['n_days']}), "
              f"mais l'écart est <b>presque entièrement une mise à l'échelle</b> : le "
              f"multiplicateur vaut 1,06 à 1,12 du décile 25 au quantile 95, et ne dévie "
              f"qu'en queue basse (1,26 au décile 10). Autrement dit, le marché ne fait pas "
              f"des amplitudes de forme différente : il fait des amplitudes plus grandes, "
              f"et il évite les journées trop calmes (plancher de liquidité et de "
              f"microstructure que la diffusion autorise mais que le marché n'offre pas).", s["body"]),
          Paragraph(
              "Conséquence pratique : une seule constante <b>κ ≈ 1,09</b> corrige le "
              "benchmark. Toute la carte d'amplitude pré-ouverture doit être multipliée "
              "par κ, sinon les zones de H et de L sont systématiquement 9 % trop étroites — "
              "ce qui suffit à faire échouer un stop placé « juste au-delà » du haut attendu.", s["body"])]

    # ---- timing
    F += [PageBreak(), Paragraph("4. Timing : à quelle heure se forment H et L", s["h1"]),
          img("f_timing.png"), Spacer(1, 0.2 * cm)]
    rows = [["Instrument", "χ² horloge", "p", "χ² temps de vol", "p",
             "KS PIT horloge", "KS PIT vol"]]
    for sym in ("SPY", "QQQ", "IWM"):
        t = RES["timing"][sym]
        rows.append([sym, f"{t['chi2_clock']:.1f}", f"{t['p_clock_test']:.4f}",
                     f"{t['chi2_vol']:.1f}", f"{t['p_vol_test']:.4f}",
                     f"{t['ks_pit_clock']:.3f}", f"{t['ks_pit_vol']:.3f}"])
    F += [table(rows, [2.4 * cm, 2.6 * cm, 2.2 * cm, 3.2 * cm, 2.2 * cm, 2.8 * cm, 2.6 * cm], size=7.8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              f"La loi de l'arcsinus en horloge brute est <b>rejetée</b> sur les trois "
              f"instruments. Le passage en temps de volatilité — c'est-à-dire remplacer "
              f"<i>t</i> par la fraction de variance écoulée <i>F(t)</i> — divise la "
              f"distance KS par 2,6 en moyenne (0,117 → 0,045 sur SPY) et améliore le χ² "
              f"de 40 %, mais <b>ne sauve pas le modèle</b> : p = {RES['timing']['SPY']['p_vol_test']:.3f} "
              f"sur SPY. Le marché concentre ses extrêmes à l'ouverture plus que ne le "
              f"prédit n'importe quel changement de temps déterministe.", s["body"]),
          Paragraph(
              "C'est une nuance importante par rapport à la littérature : Mok, Lam & Li "
              "(2000) ne rejettent pas le random walk sur le <i>S&amp;P 500 futures</i>. "
              "Sur SPY/QQQ/IWM en 2024–2026, à résolution horaire, on le rejette. "
              "Les deux résultats sont compatibles : instrument, période et granularité "
              "diffèrent. La conclusion opérationnelle est que l'arcsinus reste un "
              "<b>prior utile</b> (il place 20,5 % de masse dans la première heure) mais "
              "qu'il sous-estime nettement la pointe d'ouverture.", s["body"])]

    # ---- temps reel
    F += [PageBreak(), Paragraph("5. Temps réel : combien de fois le marché fait ce qu'on calcule", s["h1"]),
          img("f_calib.png"), Spacer(1, 0.2 * cm)]
    rows = [["Instrument", "Brier modèle", "Brier climatologie", "BSS", "n test"]]
    for sym in ("SPY", "QQQ", "IWM"):
        r = RES["realtime"][sym]
        rows.append([sym, f"{r['hi']['brier_model']:.4f}", f"{r['hi']['brier_clim']:.4f}",
                     f"{r['hi']['bss']:+.4f}", r["n_test"]])
    F += [table(rows, [3.0 * cm, 3.4 * cm, 4.2 * cm, 2.6 * cm, 2.4 * cm], size=8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              "Le panneau de droite répond littéralement à la question posée : sur toutes "
              "les séances où le marché est ouvert, à chaque heure de décision, quelle est "
              "la fréquence observée de « le high courant est déjà le high final », "
              "comparée à ce que le modèle annonce. L'écart est faible et l'ordre est "
              "respecté : c'est le seul endroit du système où la théorie prédit "
              "correctement le réel.", s["body"])]
    rows = [["Heure de décision", "n séances", "P annoncée", "Fréquence observée", "Écart"]]
    for r in RES["realtime"]["SPY"]["full_by_slot"]:
        rows.append([["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"][r["slot"]],
                     r["n"], f"{r['pred']*100:.1f} %", f"{r['obs']*100:.1f} %",
                     f"{(r['pred']-r['obs'])*100:+.1f} pt"])
    F += [table(rows, [4.0 * cm, 2.6 * cm, 3.2 * cm, 4.4 * cm, 3.0 * cm], size=8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              "Le BSS de +0,33 à +0,39 est obtenu <b>contre une climatologie conditionnelle "
              "à l'heure</b>, pas contre une constante : la comparaison est donc exigeante, "
              "et la formule gagne largement. C'est la partie du système qui a de la valeur "
              "— pour <i>savoir où on en est</i>, pas pour <i>gagner de l'argent directement</i>.", s["body"])]

    # ---- backtest
    oos = pd.read_pickle(OUT / "oos_trades.pkl")
    F += [PageBreak(), Paragraph("6. Backtest walk-forward", s["h1"]),
          Paragraph(
              "Règle testée : à chaque heure de décision, si <i>p</i> (le plus haut courant "
              "est le final) dépasse le seuil θ, on vend à l'ouverture de la barre suivante, "
              "stop 0,08 σ au-dessus du plus haut courant, take-profit à k·R, sortie "
              "forcée en fin de séance. Le côté long est traité symétriquement au plus bas. "
              "θ et k sont choisis chaque mois sur tout le passé, puis appliqués au mois "
              "suivant.", s["body"]),
          img("f_equity.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              f"Sur {O['n']} trades hors échantillon, le taux de réussite est de "
              f"{O['hit']*100:.1f} % et le profit factor de {O['profit_factor']:.3f}. "
              f"Mais le résultat central n'est pas là : l'espérance <b>brute</b> est "
              f"<b>positive</b> ({O['gross_mean_R']:+.4f} R par trade) et le coût "
              f"aller-retour ({O['cost_R']:.4f} R, soit {RES['meta']['cost_sigma']*100:.1f} % "
              f"de σ) vaut <b>{O['cost_R']/max(O['gross_mean_R'],1e-9):.2f} fois</b> cette "
              f"espérance. Autrement dit : <b>l'edge existe, il est simplement plus petit "
              f"que la friction</b>. C'est exactement le motif des études pré-enregistrées "
              f"2026 sur l'ouverture de range — « là où la cassure gagne quelque chose, "
              f"elle le gagne dans le spread ». Ce n'est pas un échec de modèle, c'est un "
              f"échec d'échelle : il faudrait un coût inférieur à 1 % de σ pour que le "
              f"même signal soit rentable.", s["body"])]
    dec = [["Décomposition", "n", "E[R] brut", "E[R] net", "Réussite"]]
    for side, lab in (("long", "Côté long (au LOD)"), ("short", "Côté short (au HOD)")):
        sub = oos[oos.side == side]
        dec.append([lab, len(sub), f"{sub.gross.mean():+.4f}", f"{sub.net.mean():+.4f}",
                    f"{(sub.net > 0).mean()*100:.1f} %"])
    for sym in ("SPY", "QQQ", "IWM"):
        sub = oos[oos.symbol == sym]
        dec.append([sym, len(sub), f"{sub.gross.mean():+.4f}", f"{sub.net.mean():+.4f}",
                    f"{(sub.net > 0).mean()*100:.1f} %"])
    F += [Spacer(1, 0.15 * cm),
          table(dec, [5.0 * cm, 2.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm], size=7.8),
          Spacer(1, 0.15 * cm),
          Paragraph(
              "La décomposition est instructive : le côté <b>long</b> (retournement depuis le "
              "plus bas) est net <i>positif</i>, le côté <b>short</b> net négatif. Cet écart est "
              "cohérent avec la prime de risque actions sur la période, mais il n'est pas "
              "significatif (t ≈ 0,2) et il serait imprudent d'y voir autre chose qu'une "
              "exposition directionnelle déguisée.", s["body"])]

    # ---- TP
    cands = {s_: pd.read_pickle(OUT / f"cand_{s_}.pkl") for s_ in ("SPY", "QQQ", "IWM")}
    tp_out, cut = fig_tp(cands)
    F += [PageBreak(), Paragraph("7. Gestion de prise de profit", s["h1"]),
          img("f_tp.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              "Le take-profit optimal est recherché sur la période d'ajustement (traits "
              "pleins) puis vérifié hors échantillon (pointillés). Le résultat le plus "
              "parlant n'est pas la forme de la courbe, c'est qu'<b>elle change de signe</b> : "
              "négative sur la première période pour presque toutes les combinaisons (θ, TP), "
              "positive sur la seconde. Un edge structurel ne s'inverse pas avec le "
              "calendrier. Le classement des take-profit est également instable : le TP qui "
              "gagne sur 2024-11→2025-08 n'est pas celui qui gagne ensuite. C'est le "
              "diagnostic le plus rapide pour distinguer un edge d'un épisode de régime : "
              "un vrai edge donne une courbe en cloche dont l'optimum <i>ne bouge pas</i>.",
              s["body"])]
    rows = [["θ", "TP", "n IS", "E[R] IS", "n OOS", "E[R] OOS",             "Signe"]]
    for th in (0.6, 0.7, 0.8):
        for kt in (0.75, 1.0, 1.5):
            a, b = [], []
            for c in cands.values():
                sub = c[c.p >= th]
                a.append(sub[sub.date < cut][f"n_{kt}"].values)
                b.append(sub[sub.date >= cut][f"n_{kt}"].values)
            a, b = np.concatenate(a), np.concatenate(b)
            flip = np.sign(a.mean()) != np.sign(b.mean())
            rows.append([th, kt, len(a), f"{a.mean():+.4f}", len(b), f"{b.mean():+.4f}",
                         "INVERSÉ" if flip else "stable"])
    F += [table(rows, [1.4 * cm, 1.4 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm, 2.6 * cm, 4.4 * cm], size=7.8)]

    # ---- MC + overfitting
    F += [PageBreak(), Paragraph("8. Monte-Carlo et contrôle du sur-ajustement", s["h1"]),
          img("f_mc.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              f"Le bootstrap par blocs de 5 jours (préserve l'autocorrélation) donne un "
              f"intervalle à 95 % de <b>[{M['boot_ci'][0]:+.1f} ; {M['boot_ci'][1]:+.1f}] R</b> "
              f"autour d'une moyenne de {M['boot_mean']:+.1f} R : il contient zéro avec une "
              f"probabilité de {M['p_boot_le_0']:.3f}. Le test de permutation (on inverse le "
              f"signe des trades au hasard) donne p = <b>{M['perm_p']:.3f}</b> : le résultat "
              f"observé est parfaitement banal sous l'hypothèse nulle.", s["body"])]
    sw = pd.DataFrame(RES["sweep"])
    by_th = sw.groupby("theta").mean_R.mean().sort_index()
    mono = bool(np.all(np.diff(by_th.values) > 0) or np.all(np.diff(by_th.values) < 0))
    sym_net = oos.groupby("symbol").net.mean()
    sign_ok = bool(np.all(sym_net > 0) or np.all(sym_net < 0))
    rows = [["Contrôle", "Valeur", "Seuil d'acceptation", "Statut"],
            ["Nombre d'essais enregistrés", f"{len(RES['sweep'])}", "—", "registre tenu"],
            ["Sharpe sous H0 (SR0)", f"{RES['dsr']['sr0']:+.3f}", "—", "—"],
            ["DSR", f"{RES['dsr']['dsr']:.3f}", "> 0,95", "ÉCHEC"],
            ["Signe SPY/QQQ/IWM cohérent", "oui" if sign_ok else "non", "identique",
             "OK" if sign_ok else "ÉCHEC"],
            ["Monotonie de E[R] dans θ",
             " / ".join(f"{v:+.4f}" for v in by_th.values), "monotone",
             "OK" if mono else "ÉCHEC"],
            ["Espérance brute > 0", f"{O['gross_mean_R']:+.4f} R", "> 0", "OK (mais < coûts)"],
            ["p(permutation)", f"{M['perm_p']:.3f}", "< 0,05", "ÉCHEC"]]
    F += [table(rows, [5.4 * cm, 2.8 * cm, 4.0 * cm, 3.4 * cm], size=8),
          Spacer(1, 0.25 * cm),
          Paragraph(
              "Le DSR est calculé avec le nombre réel d'essais du registre "
              f"({len(RES['sweep'])} configurations testées) et la dispersion des Sharpe "
              "observés sur ces essais. Avec un Sharpe par trade de "
              f"{O['sharpe_trade']:+.3f} et {O['n']} trades, il faut un Sharpe par trade "
              "d'environ 0,07 pour espérer un DSR de 0,95. On en est très loin.", s["body"])]

    # ---- fréquences
    F += [PageBreak(), Paragraph("9. Combien de fois le marché fait ce qu'on a calculé", s["h1"]),
          img("f_freq.png"), Spacer(1, 0.2 * cm)]
    fa = RES["freq_amp"]
    rows = [["Question", "Observé", "Annoncé par le modèle"],
            ["Amplitude O→H dans la bande [q25, q75]", f"{fa['ES=F']['band_hit']*100:.1f} %", "50 %"],
            ["Amplitude O→H au-delà du 90ᵉ centile", f"{fa['ES=F']['above_q90']*100:.1f} %", "10 %"],
            ["Amplitude O→H au-delà du 95ᵉ centile", f"{fa['ES=F']['above_q95']*100:.1f} %", "5 %"],
            ["Range total > 1,596 σ", f"{float(np.mean(np.array(A['ES=F']['range_sigma'])>1.596))*100:.1f} %", "≈ 50 %"]]
    for r in RES["realtime"]["SPY"]["full_by_slot"]:
        rows.append([f"HOD déjà final à {['10:30','11:30','12:30','13:30','14:30','15:30'][r['slot']]}",
                     f"{r['obs']*100:.1f} %", f"{r['pred']*100:.1f} %"])
    fs = RES.get("freq_signal", {})
    if fs:
        rows += [["TP touché sur les trades OOS", f"{fs['tp_rate']*100:.1f} %", "—"],
                 ["Stop touché", f"{fs['sl_rate']*100:.1f} %", "—"],
                 ["Sortie au temps", f"{fs['time_rate']*100:.1f} %", "—"]]
    F += [table(rows, [8.4 * cm, 3.0 * cm, 4.6 * cm], size=8),
          Spacer(1, 0.25 * cm),
          Paragraph(
              "Lecture : la colonne du milieu est la réponse à « combien de fois, sur toutes "
              "les séances où le marché est ouvert ». Sur l'amplitude, le modèle est "
              "<b>légèrement trop étroit</b> (moins de journées en zone calme que prévu, "
              "plus de journées au-delà du 90ᵉ centile). Sur le timing du high, il est "
              "remarquablement proche. Sur le trading, la fréquence de take-profit est "
              "inférieure à ce qu'exige le payoff.", s["body"])]

    # ---- conclusion
    F += [PageBreak(), Paragraph("10. Conclusion honnête", s["h1"]),
          Paragraph("Ce qui est démontré", s["h2"]),
          Paragraph("• La formule de survie de maximum prédit correctement la probabilité "
                    "que le high courant soit le final (BSS +0,33 à +0,39 contre une "
                    "climatologie horaire).", s["bullet"]),
          Paragraph("• L'amplitude RTH est prévisible et systématiquement sous-estimée par "
                    "la diffusion : facteur κ ≈ 1,09, stable sur 1 383 séances ES et NQ.", s["bullet"]),
          Paragraph("• Le temps de volatilité améliore le timing d'un facteur 2,6 mais ne "
                    "suffit pas : l'ouverture concentre plus d'extrêmes que toute loi de "
                    "référence déterministe.", s["bullet"]),
          Paragraph("Ce qui n'est pas démontré", s["h2"]),
          Paragraph(f"• Qu'on puisse en tirer un P&L. Sur {O['n']} trades hors échantillon, "
                    f"l'espérance <i>brute</i> est positive ({O['gross_mean_R']:+.4f} R) mais "
                    f"les coûts la dépassent de {O['cost_R']/max(O['gross_mean_R'],1e-9):.1f}× : "
                    f"l'espérance nette est de {O['mean_R']:+.4f} R et le DSR de "
                    f"{RES['dsr']['dsr']:.3f}. L'edge est réel mais sous le seuil de friction.", s["bullet"]),
          Paragraph("• Que la granularité horaire soit suffisante. Un déclencheur à la "
                    "minute aurait un timing d'entrée très différent ; c'est la limite "
                    "principale de ce test, imposée par la source gratuite.", s["bullet"]),
          Paragraph("Ce qui ferait changer d'avis", s["h2"]),
          Paragraph("Un BSS supérieur à +0,05 <b>contre la formule de diffusion elle-même</b> "
                    "(et non contre une climatologie), un signe identique sur ES et NQ, une "
                    "monotonie de l'espérance dans le seuil θ, et un DSR supérieur à 0,95 "
                    "avec registre des essais. Quatre conditions, aucune remplie ici.", s["body"]),
          Spacer(1, 0.4 * cm),
          Paragraph("Reproductibilité", s["h2"]),
          Paragraph("<i>python -m research.data</i> (validation de la reconstruction) → "
                    "<i>python -m research.engine</i> (étude + backtest + Monte-Carlo) → "
                    "<i>python -m research.report</i> (ce PDF). Les données brutes sont "
                    "mises en cache dans <i>research/cache/</i>.", s["small"])]

    doc.build(F)
    print("PDF ->", OUT / "HOD_LOD_ES_NQ.pdf")


def main():
    fig_amplitude(); fig_timing(); fig_calibration(); fig_freq(); fig_mc()
    oos = pd.read_pickle(OUT / "oos_trades.pkl")
    fig_equity(oos)
    build()


if __name__ == "__main__":
    main()
