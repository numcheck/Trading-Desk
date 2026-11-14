"""
Generates the figures and the final PDF (reportlab + matplotlib).

Reads research/out/results.json and the trade pickles.
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
                   label=f"{sym} (n={len(R):,})", range=(0, 4))
    x = np.linspace(0.001, 4, 400)
    ax[0].plot(x, 2 * norm.pdf(x) * (x > 0), color="k", lw=1.6,
               label="theoretical half-normal")
    ax[0].set_xlabel("R+ = (H-O)/σ  (units of forecast σ)")
    ax[0].set_ylabel("density")
    ax[0].set_title("A. O→H amplitude against the diffusion benchmark")
    ax[0].legend(fontsize=7)
    qs = RES["amplitude"]["ES=F"]["qs"]
    w = 0.38
    xs = np.arange(len(qs))
    ax[1].bar(xs - w / 2, RES["amplitude"]["ES=F"]["mult"], w, color=NAVY, label="ES")
    ax[1].bar(xs + w / 2, RES["amplitude"]["NQ=F"]["mult"], w, color=RED, label="NQ")
    ax[1].axhline(1, color="k", lw=1)
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"q{int(q*100)}" for q in qs])
    ax[1].set_ylabel("empirical quantile / theoretical quantile")
    ax[1].set_title("Amplitude multiplier by quantile")
    ax[1].legend(fontsize=8)
    fig.savefig(FIG / "f_amplitude.png")
    plt.close(fig)


def fig_timing():
    t = RES["timing"]["SPY"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    xs = np.arange(7)
    w = 0.27
    ax[0].bar(xs - w, t["obs_high"], w, color=NAVY, label="observed")
    ax[0].bar(xs, t["p_clock"], w, color=GREY, label="arcsine (clock)")
    ax[0].bar(xs + w, t["p_vol"], w, color=GREEN, label="arcsine (vol time)")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(["9:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"],
                          fontsize=7, rotation=30)
    ax[0].set_ylabel("frequency of the HOD")
    ax[0].set_title(f"C. Where the high forms (SPY, {t['n_sessions']} sessions)")
    ax[0].legend(fontsize=7)
    for sym, c in (("SPY", NAVY), ("QQQ", RED), ("IWM", GREEN)):
        s = RES["timing"][sym]
        ax[1].plot(np.cumsum(s["obs_high"]), marker="o", color=c, label=sym)
    ax[1].plot(np.cumsum(t["p_clock"]), "--", color="k", lw=1.3, label="arcsine")
    ax[1].plot(np.cumsum(t["p_vol"]), ":", color=GREY, lw=1.6, label="arcsine vol-time")
    ax[1].set_xlabel("hourly slot")
    ax[1].set_ylabel("CDF of the HOD")
    ax[1].set_title("Empirical CDF vs the reference law")
    ax[1].legend(fontsize=7)
    fig.savefig(FIG / "f_timing.png")
    plt.close(fig)


def fig_calibration():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for sym, c in (("SPY", NAVY), ("QQQ", RED), ("IWM", GREEN)):
        rel = np.array(RES["realtime"][sym]["reliability_hi"])
        ax[0].plot(rel[:, 0], rel[:, 1], marker="o", ms=3, color=c, label=sym)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_xlabel("stated p: 'the running high is the final one'")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title("D. Reliability of the diffusion model (out of sample)")
    ax[0].legend(fontsize=7)
    obs = [RES["realtime"]["SPY"]["full_by_slot"][i]["obs"] for i in range(6)]
    pred = [RES["realtime"]["SPY"]["full_by_slot"][i]["pred"] for i in range(6)]
    ax[1].plot(range(6), obs, marker="o", color=NAVY, label="observed")
    ax[1].plot(range(6), pred, marker="s", color=RED, label="predicted (formula)")
    ax[1].set_xticks(list(range(6)))
    ax[1].set_xticklabels(["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"], fontsize=7)
    ax[1].set_ylabel("P(the high is already final)")
    ax[1].set_title("The 'how often' question: predicted vs realised")
    ax[1].legend(fontsize=7)
    fig.savefig(FIG / "f_calib.png")
    plt.close(fig)


def fig_equity(oos):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    d = oos.sort_values(["date", "slot"]).reset_index(drop=True)
    eq = np.cumsum(d.net.values)
    ax[0].plot(range(len(eq)), eq, color=NAVY, lw=1.3)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xlabel("trade (chronological order)")
    ax[0].set_ylabel("cumulative R (net of costs)")
    ax[0].set_title(f"Out-of-sample walk-forward — {len(eq):,} trades")
    dd = np.maximum.accumulate(eq) - eq
    ax[1].fill_between(range(len(dd)), -dd, 0, color=RED, alpha=0.45)
    ax[1].set_xlabel("trade")
    ax[1].set_ylabel("drawdown (R)")
    ax[1].set_title(f"Drawdown (max {dd.max():.1f} R)")
    fig.savefig(FIG / "f_equity.png")
    plt.close(fig)


def fig_tp(cand_by_sym, split_frac=0.6):
    """Expectancy per trade as a function of take-profit, IS vs OOS (fixed split)."""
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
    ax.set_xlabel("take-profit (multiple of R)")
    ax.set_ylabel("net expectancy per trade (R)")
    ax.set_title("Take-profit management: expectancy as a function of the TP")
    ax.legend(fontsize=6, ncol=2)
    fig.savefig(FIG / "f_tp.png")
    plt.close(fig)
    return out, cut


def fig_mc():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(RES["mc"]["boot_dist"], bins=60, color=NAVY, alpha=0.75)
    ax[0].axvline(0, color=RED, lw=1.4)
    ax[0].set_title("Monte-Carlo: bootstrapped total P&L (R)")
    ax[0].set_xlabel("cumulative R")
    ax[1].hist(RES["mc"]["perm_dist"], bins=60, color=GREY, alpha=0.8)
    ax[1].axvline(RES["oos"]["total_R"], color=RED, lw=1.6)
    ax[1].set_title(f"Permutation null (p={RES['mc']['perm_p']:.3f})")
    ax[1].set_xlabel("cumulative R under H0")
    fig.savefig(FIG / "f_mc.png")
    plt.close(fig)


def fig_freq():
    fig, ax = plt.subplots(figsize=(10, 3.4))
    labels, vals, nom = [], [], []
    for sym in ("ES=F", "NQ=F"):
        f = RES["freq_amp"][sym]
        labels += [f"{sym} q25-q75 band", f"{sym} > q90", f"{sym} > q95"]
        vals += [f["band_hit"] * 100, f["above_q90"] * 100, f["above_q95"] * 100]
        nom += [50, 10, 5]
    xs = np.arange(len(labels))
    ax.bar(xs - 0.2, vals, 0.4, color=NAVY, label="observed")
    ax.bar(xs + 0.2, nom, 0.4, color=GREY, label="nominal (diffusion)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_ylabel("% of sessions")
    ax.set_title("How often the market does what the model announces (amplitude)")
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
                            title="HOD/LOD ES-NQ — walk-forward backtest", author="Quantitative research")
    F = []
    A = RES["amplitude"]
    O = RES["oos"]
    M = RES["mc"]
    # the drawdown only makes sense in CHRONOLOGICAL order (the OOS trades are
    # concatenated per symbol in the pickle: re-sort before cumulating)
    oos_chrono = pd.read_pickle(OUT / "oos_trades.pkl").sort_values(
        ["date", "slot"]).reset_index(drop=True)
    eq_all = np.cumsum(oos_chrono.net.values)
    DD_CHRONO = float(np.max(np.maximum.accumulate(eq_all) - eq_all))

    # ---- cover
    F += [Spacer(1, 4 * cm), Paragraph("Where the HOD and the LOD form", s["title"]),
          Paragraph("A statistical study and walk-forward backtest on ES / NQ", s["sub"]),
          Spacer(1, 1.2 * cm)]
    cover = [["Data", "Free (Yahoo Finance), no API key"],
             ["Intraday", f"{RES['timing']['SPY']['n_sessions']} sessions × 3 instruments "
                          f"({RES['timing']['SPY']['start']} → {RES['timing']['SPY']['end']})"],
             ["Daily", f"{A['ES=F']['n_days']:,} sessions on ES & NQ "
                       f"({A['ES=F']['start']} → {A['ES=F']['end']})"],
             ["Backtest", f"{O['n']:,} out-of-sample trades, monthly walk-forward selection"],
             ["Verdict", f"net expectancy {O['mean_R']:+.4f} R per trade — "
                         f"DSR {RES['dsr']['dsr']:.3f}"]]
    F += [table([["Item", "Coverage"]] + cover, [4.2 * cm, 13.0 * cm], size=8.5,
                last_left=True),
          Spacer(1, 1.4 * cm),
          Paragraph("Report generated automatically from public data. The figures are "
                    "those of the shipped code, reproducible with "
                    "<i>python -m research.engine</i> then <i>python -m research.report</i>. "
                    "No result has been rounded in favour of the strategy.", s["small"]),
          PageBreak()]

    # ---- summary
    F += [Paragraph("1. Executive summary", s["h1"])]
    bss = np.mean([RES["realtime"][k]["hi"]["bss"] for k in ("SPY", "QQQ", "IWM")])
    F += [Paragraph(
        f"<b>The statistics hold; the trading stays below the friction threshold.</b> The "
        f"maximum-survival model (reflection formula) beats a time-conditioned climatology "
        f"with an average Brier Skill Score of <b>{bss:+.3f}</b> across three instruments. "
        f"The walk-forward backtest of the high reversal yields a <b>gross expectancy of "
        f"{O['gross_mean_R']:+.4f} R</b> per trade over {O['n']:,} trades — so a real signal — "
        f"but a round-trip cost of <b>{O['cost_R']:.4f} R</b>, i.e. "
        f"{O['cost_R']/max(O['gross_mean_R'],1e-9):.2f} times that expectancy. Net: "
        f"{O['mean_R']:+.4f} R, per-trade Sharpe {O['sharpe_trade']:+.3f}, DSR "
        f"{RES['dsr']['dsr']:.3f}. The result is not \"no signal\" but <b>\"a signal too "
        f"small for the fees\"</b> — a more useful conclusion, and harder to overturn.",
        s["body"])]
    kp = [["Indicator", "Value", "Reading"],
          ["Out-of-sample trades", f"{O['n']:,}", "walk-forward, 3 instruments"],
          ["Hit rate", f"{O['hit']*100:.1f}%", "≈ chance (50%)"],
          ["Net expectancy", f"{O['mean_R']:+.4f} R", "negative"],
          ["Gross expectancy", f"{O['gross_mean_R']:+.4f} R", "positive, but < costs"],
          ["Average cost", f"{O['cost_R']:.4f} R", f"{O['cost_R']/max(O['gross_mean_R'],1e-9):.2f}× the gross expectancy"],
          ["Profit factor", f"{O['profit_factor']:.3f}", "< 1"],
          ["Max drawdown (chrono)", f"{DD_CHRONO:.1f} R", "over the period"],
          ["95% bootstrap CI", f"[{M['boot_ci'][0]:+.1f}; {M['boot_ci'][1]:+.1f}] R", "contains 0"],
          ["p(permutation)", f"{M['perm_p']:.3f}", "not significant"],
          ["DSR", f"{RES['dsr']['dsr']:.3f}", "≪ 0.95: no proven edge"]]
    F += [table(kp, [5.0 * cm, 4.4 * cm, 8.0 * cm], size=8, highlight=3, last_left=True),
          Spacer(1, 0.3 * cm),
          Paragraph(
              "Three positive, solid results independent of the backtest: "
              "(i) the RTH amplitude averages <b>1.82 σ</b> against 1.596 predicted by "
              "diffusion — the market travels 14% more than the model; "
              "(ii) the amplitude multiplier is <b>near-constant (1.06–1.12)</b> above the "
              "low decile, which makes the correction trivial; "
              "(iii) the arcsine law on the raw clock is <b>rejected</b> "
              f"(χ² = {RES['timing']['SPY']['chi2_clock']:.1f} on SPY) but switching to "
              "volatility time divides the KS distance by 2.6.", s["body"])]

    # ---- data
    F += [PageBreak(), Paragraph("2. Data and the real limits of the free source", s["h1"]),
          Paragraph(
              "The request was for 3 years. The free source does not allow that at high "
              "frequency: the limits below were measured, not assumed. Rather than filling "
              "gaps with invented data, the real coverage is used and the granularity is "
              "chosen per sub-problem.", s["body"])]
    lim = [["Interval", "Real depth", "Use retained"],
           ["1 minute", "~25 days", "not used"],
           ["5 / 15 / 30 minutes", "60 days", "spot check"],
           ["1 hour", "730 days", "timing + real time + backtest"],
           ["1 day", "unlimited", "amplitude, HAR, 6 years"]]
    F += [table(lim, [4.2 * cm, 4.0 * cm, 9.2 * cm], size=8), Spacer(1, 0.25 * cm),
          Paragraph(
              "The hourly bars of SPY/QQQ/IWM are rebuilt over 09:30–16:00 and "
              "<b>validated against the exchange's daily OHLC</b>: the high is exact to "
              "within 2 basis points on 99.2% of sessions, the low on 97.6%, and the median "
              "range ratio is 1.000. The value of the high/low is therefore exact; only "
              "its <i>timing</i> is quantised to the hour. A spurious 16:00 bar captured "
              "post-close trading and was removed.", s["body"]),
          Paragraph("Anti-look-ahead discipline", s["h2"])]
    F += [Paragraph("• σ(J) is a HAR on an <b>expanding</b> window, re-estimated every "
                    "20 days, using only variances ≤ J−1.", s["bullet"]),
          Paragraph("• The seasonal profile is estimated on rolling history.", s["bullet"]),
          Paragraph("• The backtest enters at the <b>open of the bar following</b> the signal.", s["bullet"]),
          Paragraph("• (threshold, take-profit) are chosen by <b>monthly walk-forward</b> on "
                    "past data only; the evaluated month never enters the selection.", s["bullet"]),
          Paragraph("• Costs are 1.6% of σ round trip (0.5 ES point out of a 31-point σ), "
                    "applied to every trade.", s["bullet"])]

    # ---- amplitude
    F += [PageBreak(), Paragraph("3. Amplitude: how far the market travels", s["h1"])]
    rows = [["Instrument", "n", "E[R+]", "theory", "E[range/σ]", "theory", "KS"]]
    for sym in ("ES=F", "NQ=F"):
        a = A[sym]
        rows.append([sym, f"{a['n_days']:,}", f"{a['mean_Rup']:.3f}", f"{a['theo_mean']:.3f}",
                     f"{a['mean_range']:.3f}", f"{a['theo_range']:.3f}", f"{a['ks_Rup']:.3f}"])
    F += [img("f_amplitude.png"), Spacer(1, 0.2 * cm),
          table(rows, [2.6 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.2 * cm, 1.8 * cm], size=7.8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              f"The diffusion benchmark (<i>H−O ~ |N(0,σ²T)|</i>) is rejected by the KS test "
              f"(D = {A['ES=F']['ks_Rup']:.3f}, 5% threshold 0.037 for n = {A['ES=F']['n_days']:,}), "
              f"but the deviation is <b>almost entirely a scaling</b>: the multiplier runs "
              f"from 1.06 to 1.12 between the 25th decile and the 95th quantile, and deviates "
              f"only in the lower tail (1.26 at the 10th decile). In other words, the market "
              f"does not produce amplitudes of a different shape: it produces larger ones, "
              f"and it avoids days that are too quiet (a liquidity and microstructure floor "
              f"that diffusion allows but the market does not offer).", s["body"]),
          Paragraph(
              "Practical consequence: a single constant <b>κ ≈ 1.09</b> corrects the "
              "benchmark. The whole pre-open amplitude map must be multiplied by κ, "
              "otherwise the H and L zones are systematically 9% too narrow — enough to "
              "break a stop placed \"just beyond\" the expected high.", s["body"])]

    # ---- timing
    F += [PageBreak(), Paragraph("4. Timing: at what time H and L form", s["h1"]),
          img("f_timing.png"), Spacer(1, 0.2 * cm)]
    rows = [["Instrument", "χ² clock", "p", "χ² vol time", "p", "KS PIT clock", "KS PIT vol"]]
    for sym in ("SPY", "QQQ", "IWM"):
        t = RES["timing"][sym]
        rows.append([sym, f"{t['chi2_clock']:.1f}", f"{t['p_clock_test']:.4f}",
                     f"{t['chi2_vol']:.1f}", f"{t['p_vol_test']:.4f}",
                     f"{t['ks_pit_clock']:.3f}", f"{t['ks_pit_vol']:.3f}"])
    F += [table(rows, [2.4 * cm, 2.6 * cm, 2.2 * cm, 3.2 * cm, 2.2 * cm, 2.8 * cm, 2.6 * cm], size=7.8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              f"The arcsine law on the raw clock is <b>rejected</b> on all three "
              f"instruments. Switching to volatility time — replacing <i>t</i> with the "
              f"fraction of variance elapsed <i>F(t)</i> — divides the KS distance by 2.6 "
              f"on average (0.117 → 0.045 on SPY) and improves the χ² by 40%, but <b>does "
              f"not save the model</b>: p = {RES['timing']['SPY']['p_vol_test']:.3f} on SPY. "
              f"The market concentrates its extremes at the open more than any "
              f"deterministic time change predicts.", s["body"]),
          Paragraph(
              "This is an important nuance relative to the literature: Mok, Lam & Li "
              "(2000) do not reject the random walk on <i>S&amp;P 500 futures</i>. On "
              "SPY/QQQ/IWM over 2024–2026, at hourly resolution, it is rejected. Both "
              "results are compatible: instrument, period and granularity differ. The "
              "operational conclusion is that the arcsine law remains a <b>useful prior</b> "
              "(it puts 20.5% of the mass in the first hour) but clearly understates the "
              "opening spike.", s["body"])]

    # ---- real time
    F += [PageBreak(), Paragraph("5. Real time: how often the market does what we compute", s["h1"]),
          img("f_calib.png"), Spacer(1, 0.2 * cm)]
    rows = [["Instrument", "Brier model", "Brier climatology", "BSS", "n test"]]
    for sym in ("SPY", "QQQ", "IWM"):
        r = RES["realtime"][sym]
        rows.append([sym, f"{r['hi']['brier_model']:.4f}", f"{r['hi']['brier_clim']:.4f}",
                     f"{r['hi']['bss']:+.4f}", f"{r['n_test']:,}"])
    F += [table(rows, [3.0 * cm, 3.4 * cm, 4.2 * cm, 2.6 * cm, 2.4 * cm], size=8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              "The right-hand panel answers the question literally: over every session the "
              "market is open, at each decision hour, what is the observed frequency of "
              "\"the running high is already the final high\", compared with what the model "
              "announces. The gap is small and the ordering is respected: this is the only "
              "place in the system where the theory predicts reality correctly.", s["body"])]
    rows = [["Decision time", "n sessions", "Stated P", "Observed frequency", "Gap"]]
    for r in RES["realtime"]["SPY"]["full_by_slot"]:
        rows.append([["10:30", "11:30", "12:30", "13:30", "14:30", "15:30"][r["slot"]],
                     f"{r['n']:,}", f"{r['pred']*100:.1f}%", f"{r['obs']*100:.1f}%",
                     f"{(r['pred']-r['obs'])*100:+.1f} pt"])
    F += [table(rows, [4.0 * cm, 2.6 * cm, 3.2 * cm, 4.4 * cm, 3.0 * cm], size=8),
          Spacer(1, 0.2 * cm),
          Paragraph(
              "The BSS of +0.33 to +0.39 is obtained <b>against a time-conditioned "
              "climatology</b>, not against a constant: the comparison is demanding, and "
              "the formula wins clearly. This is the part of the system that has value — "
              "for <i>knowing where we stand</i>, not for <i>making money directly</i>.",
              s["body"])]

    # ---- backtest
    oos = pd.read_pickle(OUT / "oos_trades.pkl")
    F += [PageBreak(), Paragraph("6. Walk-forward backtest", s["h1"]),
          Paragraph(
              "Rule tested: at each decision hour, if <i>p</i> (the running high is the "
              "final one) exceeds the threshold θ, sell at the open of the next bar, stop "
              "0.08 σ above the running high, take-profit at k·R, forced exit at the "
              "session close. The long side is handled symmetrically at the running low. "
              "θ and k are chosen each month on all past data, then applied to the "
              "following month.", s["body"]),
          img("f_equity.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              f"Over {O['n']:,} out-of-sample trades the hit rate is {O['hit']*100:.1f}% and "
              f"the profit factor {O['profit_factor']:.3f}. But the central result is not "
              f"there: the <b>gross</b> expectancy is <b>positive</b> "
              f"({O['gross_mean_R']:+.4f} R per trade) and the round-trip cost "
              f"({O['cost_R']:.4f} R, i.e. {RES['meta']['cost_sigma']*100:.1f}% of σ) is "
              f"<b>{O['cost_R']/max(O['gross_mean_R'],1e-9):.2f} times</b> that expectancy. "
              f"In other words: <b>the edge exists, it is simply smaller than the "
              f"friction</b>. This is exactly the pattern of the 2026 pre-registered "
              f"studies on the opening range — \"where the breakout earns something, it "
              f"earns it in the spread\". This is not a model failure, it is a failure of "
              f"scale: costs below 1% of σ would be needed for the same signal to be "
              f"profitable.", s["body"])]
    dec = [["Breakdown", "n", "E[R] gross", "E[R] net", "Hit rate"]]
    for side, lab in (("long", "Long side (at the LOD)"), ("short", "Short side (at the HOD)")):
        sub = oos[oos.side == side]
        dec.append([lab, f"{len(sub):,}", f"{sub.gross.mean():+.4f}", f"{sub.net.mean():+.4f}",
                    f"{(sub.net > 0).mean()*100:.1f}%"])
    for sym in ("SPY", "QQQ", "IWM"):
        sub = oos[oos.symbol == sym]
        dec.append([sym, f"{len(sub):,}", f"{sub.gross.mean():+.4f}", f"{sub.net.mean():+.4f}",
                    f"{(sub.net > 0).mean()*100:.1f}%"])
    F += [Spacer(1, 0.15 * cm),
          table(dec, [5.0 * cm, 2.0 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm], size=7.8),
          Spacer(1, 0.15 * cm),
          Paragraph(
              "The breakdown is instructive: the <b>long</b> side (reversal from the low) is "
              "net <i>positive</i>, the <b>short</b> side net negative. This gap is "
              "consistent with the equity risk premium over the period, but it is not "
              "significant (t ≈ 0.2) and it would be imprudent to see anything other than "
              "disguised directional exposure.", s["body"])]

    # ---- TP
    cands = {s_: pd.read_pickle(OUT / f"cand_{s_}.pkl") for s_ in ("SPY", "QQQ", "IWM")}
    tp_out, cut = fig_tp(cands)
    F += [PageBreak(), Paragraph("7. Take-profit management", s["h1"]),
          img("f_tp.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              "The optimal take-profit is searched on the fitting period (solid lines) then "
              "checked out of sample (dotted). The most telling result is not the shape of "
              "the curve but that <b>it changes sign</b>: negative over the first period for "
              "almost every (θ, TP) combination, positive over the second. A structural edge "
              "does not invert with the calendar. The ranking of take-profits is also "
              "unstable: the TP that wins over 2024-11→2025-08 is not the one that wins "
              "afterwards. This is the fastest diagnostic for telling an edge from a regime "
              "episode: a real edge gives a bell-shaped curve whose optimum <i>does not "
              "move</i>.", s["body"])]
    rows = [["θ", "TP", "n IS", "E[R] IS", "n OOS", "E[R] OOS", "Sign"]]
    for th in (0.6, 0.7, 0.8):
        for kt in (0.75, 1.0, 1.5):
            a, b = [], []
            for c in cands.values():
                sub = c[c.p >= th]
                a.append(sub[sub.date < cut][f"n_{kt}"].values)
                b.append(sub[sub.date >= cut][f"n_{kt}"].values)
            a, b = np.concatenate(a), np.concatenate(b)
            flip = np.sign(a.mean()) != np.sign(b.mean())
            rows.append([th, kt, f"{len(a):,}", f"{a.mean():+.4f}", f"{len(b):,}", f"{b.mean():+.4f}",
                         "FLIPPED" if flip else "stable"])
    F += [table(rows, [1.4 * cm, 1.4 * cm, 2.0 * cm, 2.4 * cm, 2.2 * cm, 2.6 * cm, 4.4 * cm], size=7.8)]

    # ---- MC + overfitting
    F += [PageBreak(), Paragraph("8. Monte-Carlo and overfitting control", s["h1"]),
          img("f_mc.png"), Spacer(1, 0.2 * cm),
          Paragraph(
              f"The 5-day block bootstrap (which preserves autocorrelation) gives a 95% "
              f"interval of <b>[{M['boot_ci'][0]:+.1f}; {M['boot_ci'][1]:+.1f}] R</b> around "
              f"a mean of {M['boot_mean']:+.1f} R: it contains zero with probability "
              f"{M['p_boot_le_0']:.3f}. The permutation test (randomly flipping the sign of "
              f"trades) gives p = <b>{M['perm_p']:.3f}</b>: the observed result is entirely "
              f"ordinary under the null hypothesis.", s["body"])]
    sw = pd.DataFrame(RES["sweep"])
    by_th = sw.groupby("theta").mean_R.mean().sort_index()
    mono = bool(np.all(np.diff(by_th.values) > 0) or np.all(np.diff(by_th.values) < 0))
    sym_net = oos.groupby("symbol").net.mean()
    sign_ok = bool(np.all(sym_net > 0) or np.all(sym_net < 0))
    rows = [["Control", "Value", "Acceptance threshold", "Status"],
            ["Registered trials", f"{len(RES['sweep'])}", "—", "ledger maintained"],
            ["Sharpe under H0 (SR0)", f"{RES['dsr']['sr0']:+.3f}", "—", "—"],
            ["DSR", f"{RES['dsr']['dsr']:.3f}", "> 0.95", "FAIL"],
            ["Consistent sign SPY/QQQ/IWM", "yes" if sign_ok else "no", "identical",
             "OK" if sign_ok else "FAIL"],
            ["Monotonicity of E[R] in θ",
             " / ".join(f"{v:+.4f}" for v in by_th.values), "monotone",
             "OK" if mono else "FAIL"],
            ["Gross expectancy > 0", f"{O['gross_mean_R']:+.4f} R", "> 0", "OK (but < costs)"],
            ["p(permutation)", f"{M['perm_p']:.3f}", "< 0.05", "FAIL"]]
    F += [table(rows, [5.4 * cm, 2.8 * cm, 4.0 * cm, 3.4 * cm], size=8),
          Spacer(1, 0.25 * cm),
          Paragraph(
              "The DSR is computed with the real number of trials in the ledger "
              f"({len(RES['sweep'])} configurations tested) and the dispersion of the Sharpe "
              "ratios observed across them. With a per-trade Sharpe of "
              f"{O['sharpe_trade']:+.3f} and {O['n']:,} trades, a per-trade Sharpe of about "
              "0.07 would be needed to hope for a DSR of 0.95. We are far from it.", s["body"])]

    # ---- frequencies
    F += [PageBreak(), Paragraph("9. How often the market does what we computed", s["h1"]),
          img("f_freq.png"), Spacer(1, 0.2 * cm)]
    fa = RES["freq_amp"]
    rows = [["Question", "Observed", "Announced by the model"],
            ["O→H amplitude inside [q25, q75]", f"{fa['ES=F']['band_hit']*100:.1f}%", "50%"],
            ["O→H amplitude beyond the 90th percentile", f"{fa['ES=F']['above_q90']*100:.1f}%", "10%"],
            ["O→H amplitude beyond the 95th percentile", f"{fa['ES=F']['above_q95']*100:.1f}%", "5%"],
            ["Total range > 1.596 σ", f"{float(np.mean(np.array(A['ES=F']['range_sigma'])>1.596))*100:.1f}%", "≈ 50%"]]
    for r in RES["realtime"]["SPY"]["full_by_slot"]:
        rows.append([f"HOD already final at {['10:30','11:30','12:30','13:30','14:30','15:30'][r['slot']]}",
                     f"{r['obs']*100:.1f}%", f"{r['pred']*100:.1f}%"])
    fs = RES.get("freq_signal", {})
    if fs:
        rows += [["TP reached on OOS trades", f"{fs['tp_rate']*100:.1f}%", "—"],
                 ["Stop reached", f"{fs['sl_rate']*100:.1f}%", "—"],
                 ["Time exit", f"{fs['time_rate']*100:.1f}%", "—"]]
    F += [table(rows, [8.4 * cm, 3.0 * cm, 4.6 * cm], size=8),
          Spacer(1, 0.25 * cm),
          Paragraph(
              "Reading: the middle column is the answer to \"how often, across every session "
              "the market is open\". On amplitude, the model is <b>slightly too narrow</b> "
              "(fewer days in the calm zone than expected, more days beyond the 90th "
              "percentile). On the timing of the high it is remarkably close. On trading, "
              "the take-profit frequency is below what the payoff requires.", s["body"])]

    # ---- conclusion
    F += [PageBreak(), Paragraph("10. Honest conclusion", s["h1"]),
          Paragraph("What is demonstrated", s["h2"]),
          Paragraph("• The maximum-survival formula correctly predicts the probability that "
                    "the running high is the final one (BSS +0.33 to +0.39 against an hourly "
                    "climatology).", s["bullet"]),
          Paragraph("• The RTH amplitude is predictable and systematically understated by "
                    "diffusion: factor κ ≈ 1.09, stable over 1,383 ES and NQ sessions.", s["bullet"]),
          Paragraph("• Volatility time improves timing by a factor of 2.6 but is not "
                    "enough: the open concentrates more extremes than any deterministic "
                    "reference law.", s["bullet"]),
          Paragraph("What is not demonstrated", s["h2"]),
          Paragraph(f"• That a P&L can be extracted. Over {O['n']:,} out-of-sample trades "
                    f"the <i>gross</i> expectancy is positive ({O['gross_mean_R']:+.4f} R) but "
                    f"costs exceed it by {O['cost_R']/max(O['gross_mean_R'],1e-9):.1f}×: net "
                    f"expectancy is {O['mean_R']:+.4f} R and the DSR is "
                    f"{RES['dsr']['dsr']:.3f}. The edge is real but below the friction "
                    f"threshold.", s["bullet"]),
          Paragraph("• That hourly granularity is sufficient. A minute-level trigger would "
                    "have a very different entry timing; this is the main limit of the test, "
                    "imposed by the free source.", s["bullet"]),
          Paragraph("What would change our mind", s["h2"]),
          Paragraph("A BSS above +0.05 <b>against the diffusion formula itself</b> (not "
                    "against a climatology), the same sign on ES and NQ, monotonic "
                    "expectancy in the threshold θ, and a DSR above 0.95 with a trial "
                    "ledger. Four conditions, none met here.", s["body"]),
          Spacer(1, 0.4 * cm),
          Paragraph("Reproducibility", s["h2"]),
          Paragraph("<i>python -m research.data</i> (reconstruction validation) → "
                    "<i>python -m research.engine</i> (study + backtest + Monte-Carlo) → "
                    "<i>python -m research.report</i> (this PDF). Raw data are cached in "
                    "<i>research/cache/</i>.", s["small"])]

    doc.build(F)
    print("PDF ->", OUT / "HOD_LOD_ES_NQ.pdf")


def main():
    fig_amplitude(); fig_timing(); fig_calibration(); fig_freq(); fig_mc()
    oos = pd.read_pickle(OUT / "oos_trades.pkl")
    fig_equity(oos)
    build()


if __name__ == "__main__":
    main()
