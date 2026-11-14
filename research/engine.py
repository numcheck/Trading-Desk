"""
Study and backtest engine on real free data.

No look-ahead:
  * sigma_hat(J) = HAR on an EXPANDING window, re-estimated every 20 days, using
    only variances <= J-1.
  * the seasonal profile is estimated on rolling history.
  * the backtest enters at the OPEN of the bar following the signal.
  * (theta, take-profit) are chosen on past data only, by monthly walk-forward.
    The evaluated period never selects.

Speed: every entry candidate (date, slot, side) is simulated ONCE for each
take-profit; filtering by threshold is then a simple mask.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, kstest, norm

from hodlod.evaluate import block_bootstrap_ci
from research.data import daily, sessions_1h

warnings.filterwarnings("ignore")
OUT = Path("research/out")
OUT.mkdir(parents=True, exist_ok=True)

COST_SIGMA = 0.016          # ES round-trip cost as a fraction of sigma_day
N_SLOTS = 7
MAX_PER_DAY = 1             # max entries per instrument per session (configurable)


# ---------------------------------------------------------------- volatility
def parkinson_var(d: pd.DataFrame) -> np.ndarray:
    return np.log(d.high.values / d.low.values) ** 2 / (4 * np.log(2))


def har_walkforward(var: np.ndarray, min_train: int = 120, step: int = 20) -> np.ndarray:
    """sigma_hat[t] strictly causal (HAR on log variance)."""
    x = np.log(np.maximum(var, 1e-12))
    s = pd.Series(x)
    X = np.column_stack([s.shift(1), s.shift(1).rolling(5).mean(),
                         s.shift(1).rolling(22).mean()])
    ok = np.isfinite(X).all(axis=1) & np.isfinite(x)
    out = np.full(len(x), np.nan)
    coef = None
    for t in range(len(x)):
        if not ok[t]:
            continue
        if coef is None or (t - min_train) % step == 0:
            if t >= min_train:
                m = ok.copy()
                m[t:] = False
                if m.sum() >= 80:
                    A = np.column_stack([np.ones(int(m.sum())), X[m]])
                    coef, *_ = np.linalg.lstsq(A, x[m], rcond=None)
        if coef is not None and t > min_train:
            out[t] = np.exp(0.5 * (coef[0] + X[t] @ coef[1:]))
    return out


def profile_slots(bars: pd.DataFrame) -> np.ndarray:
    r = np.log(bars.close.values / bars.open.values)
    v = (pd.DataFrame({"slot": bars.slot.values, "r2": r ** 2})
         .groupby("slot").r2.mean().reindex(range(N_SLOTS)).ffill().bfill().values)
    return v / v.sum()


def _arc(x):
    return (2 / np.pi) * np.arcsin(np.sqrt(np.clip(x, 0, 1)))


# ---------------------------------------------------------------- A. amplitude
def study_amplitude(sym: str, years: int = 6) -> dict:
    d = daily(sym, years)
    # har_walkforward returns a sigma in LOG-price: convert it back to points.
    # The conversion factor is the CURRENT OPEN (known at 09:30). Multiplying by the
    # same day's close leaks the close into a quantity used to size the day.
    sig = har_walkforward(parkinson_var(d)) * d.open.values
    m = np.isfinite(sig) & (sig > 0)
    d, sig = d[m], sig[m]
    R_up = (d.high.values - d.open.values) / sig
    R_dn = (d.open.values - d.low.values) / sig
    rng = (d.high.values - d.low.values) / sig
    qs = np.array([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    emp = np.quantile(R_up, qs)
    th = norm.ppf(0.5 + 0.5 * qs)
    ks = kstest(R_up, lambda z: 2 * norm.cdf(z) - 1)
    return {
        "symbol": sym, "n_days": int(len(d)),
        "start": str(d.index.min().date()), "end": str(d.index.max().date()),
        "R_up": R_up.tolist(), "R_dn": R_dn.tolist(), "range_sigma": rng.tolist(),
        "qs": qs.tolist(), "mult": (emp / th).tolist(),
        "mean_Rup": float(np.mean(R_up)), "theo_mean": float(np.sqrt(2 / np.pi)),
        "mean_Rdn": float(np.mean(R_dn)),
        "mean_range": float(np.mean(rng)), "theo_range": float(np.sqrt(8 / np.pi)),
        "ks_Rup": float(ks.statistic), "ks_p": float(ks.pvalue),
        "med_sigma_pts": float(np.median(sig)),
        "dates": [str(x.date()) for x in d.index],
    }


def frequency_amplitude(a: dict) -> dict:
    R = np.array(a["R_up"])
    lo, hi = norm.ppf(0.625), norm.ppf(0.875)
    return {"band_hit": float(np.mean((R >= lo) & (R <= hi))), "band_nominal": 0.50,
            "above_q90": float(np.mean(R > norm.ppf(0.95))), "q90_nominal": 0.10,
            "above_q95": float(np.mean(R > norm.ppf(0.975))), "q95_nominal": 0.05,
            "above_median": float(np.mean(R > norm.ppf(0.75))), "median_nominal": 0.50}


# ---------------------------------------------------------------- C. timing
def study_timing(sym: str, days: int = 730) -> dict:
    bars, sess = sessions_1h(sym, days)
    prof = profile_slots(bars)
    F = np.cumsum(prof)
    edges = np.arange(N_SLOTS + 1) / N_SLOTS
    p_clock = np.diff(_arc(edges))
    p_vol = np.diff(_arc(np.r_[0.0, F]))
    cnt = np.bincount(sess.i_high.values, minlength=N_SLOTS).astype(float)
    n = len(sess)
    obs = cnt / n
    chi_c = float(np.sum((cnt - n * p_clock) ** 2 / np.maximum(n * p_clock, 1e-9)))
    chi_v = float(np.sum((cnt - n * p_vol) ** 2 / np.maximum(n * p_vol, 1e-9)))
    idx = sess.i_high.values
    # Randomised PIT: on a DISCRETE variable with 7 modalities the raw KS is
    # invalid (step CDF). We add a uniform jitter inside the slot.
    rng = np.random.default_rng(0)
    e_clock = _arc(edges)
    e_vol = _arc(np.r_[0.0, F])
    ks_c, ks_v = [], []
    for _ in range(20):
        u_c = e_clock[idx] + rng.uniform(size=n) * (e_clock[idx + 1] - e_clock[idx])
        u_v = e_vol[idx] + rng.uniform(size=n) * (e_vol[idx + 1] - e_vol[idx])
        ks_c.append(kstest(u_c, "uniform").statistic)
        ks_v.append(kstest(u_v, "uniform").statistic)
    ls = lambda p: float(-np.mean(np.log(np.maximum(np.asarray(p)[idx], 1e-12))))
    return {
        "symbol": sym, "n_sessions": int(n),
        "start": str(sess.index.min().date()), "end": str(sess.index.max().date()),
        "profile": prof.tolist(), "F": F.tolist(), "obs_high": obs.tolist(),
        "p_clock": p_clock.tolist(), "p_vol": p_vol.tolist(),
        "chi2_clock": chi_c, "p_clock_test": float(1 - chi2.cdf(chi_c, N_SLOTS - 1)),
        "chi2_vol": chi_v, "p_vol_test": float(1 - chi2.cdf(chi_v, N_SLOTS - 1)),
        "ks_pit_clock": float(np.mean(ks_c)), "ks_pit_vol": float(np.mean(ks_v)),
        "logscore_clock": ls(p_clock), "logscore_vol": ls(p_vol),
        "logscore_emp": float(-np.mean(np.log(np.maximum(obs, 1e-12)))),
        "first_hour_obs": float(obs[0]), "first_hour_theo": float(p_clock[0]),
        "last2_obs": float(obs[5] + obs[6]), "last2_theo": float(p_clock[5] + p_clock[6]),
        "t_high": sess.t_high.tolist(),
        "obs_low": (np.bincount(sess.i_low.values, minlength=N_SLOTS) / n).tolist(),
    }


# ---------------------------------------------------------------- D. real time
def realtime_rows(sym: str, days: int = 730) -> pd.DataFrame:
    bars, sess = sessions_1h(sym, days)
    prof = profile_slots(bars)
    F = np.cumsum(prof)
    # log-vol -> points with the current open (known at 09:30); the same day's
    # close would be look-ahead
    sig = har_walkforward(parkinson_var(sess)) * sess.open.values   # in points
    pos = {d: i for i, d in enumerate(sess.index)}
    rows = []
    for day, g in bars.groupby("date"):
        if day not in pos:
            continue
        i = pos[day]
        if not np.isfinite(sig[i]):
            continue
        g = g.sort_values("slot")
        if len(g) < 6:
            continue
        s = float(sig[i])
        hi, lo, cl = g.high.values, g.low.values, g.close.values
        fh, fl = hi.max(), lo.min()
        rh, rl = -np.inf, np.inf
        for k in range(len(g) - 1):
            rh, rl = max(rh, hi[k]), min(rl, lo[k])
            S = s * np.sqrt(max(1 - F[k], 1e-6))
            rows.append({"date": day, "slot": k, "sigma": s, "F": F[k],
                         "p_hi": float(2 * norm.cdf((rh - cl[k]) / S) - 1),
                         "p_lo": float(2 * norm.cdf((cl[k] - rl) / S) - 1),
                         "d_hi": (rh - cl[k]) / s, "d_lo": (cl[k] - rl) / s,
                         "y_hi": int(rh >= fh - 1e-12), "y_lo": int(rl <= fl + 1e-12)})
    return pd.DataFrame(rows)


def brier_table(rt: pd.DataFrame, train_frac: float = 0.6) -> dict:
    days = np.array(sorted(rt.date.unique()))
    cut = days[int(len(days) * train_frac)]
    tr, te = rt[rt.date < cut], rt[rt.date >= cut]
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "cut": str(cut)}
    for side in ("hi", "lo"):
        base = tr.groupby("slot")[f"y_{side}"].mean()
        pb = te.slot.map(base).values
        bs_m = float(np.mean((te[f"p_{side}"].values - te[f"y_{side}"].values) ** 2))
        bs_b = float(np.mean((pb - te[f"y_{side}"].values) ** 2))
        out[side] = {"brier_model": bs_m, "brier_clim": bs_b,
                     "bss": float(1 - bs_m / bs_b) if bs_b > 0 else np.nan,
                     "base_rate": float(te[f"y_{side}"].mean())}
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(te.p_hi.values, bins) - 1, 0, 9)
    out["reliability_hi"] = [[float(te.p_hi.values[idx == b].mean()),
                              float(te.y_hi.values[idx == b].mean()),
                              int((idx == b).sum())] for b in range(10) if (idx == b).sum() > 5]
    out["by_slot"] = [{"slot": int(k), "n": int((te.slot == k).sum()),
                       "obs": float(te[te.slot == k].y_hi.mean()),
                       "pred": float(te[te.slot == k].p_hi.mean())}
                      for k in sorted(te.slot.unique())]
    out["full_by_slot"] = [{"slot": int(k), "n": int((rt.slot == k).sum()),
                            "obs": float(rt[rt.slot == k].y_hi.mean()),
                            "pred": float(rt[rt.slot == k].p_hi.mean())}
                           for k in sorted(rt.slot.unique())]
    return out


# ---------------------------------------------------------------- backtest
def simulate(rt: pd.DataFrame, bars: pd.DataFrame, k_tps, max_risk_sigma: float = 0.7,
             buffer_sigma: float = 0.08) -> pd.DataFrame:
    """Simulates every entry candidate, for each take-profit."""
    by_day = {d: g.sort_values("slot") for d, g in bars.groupby("date")}
    recs = []
    for row in rt.itertuples():
        g = by_day.get(row.date)
        if g is None or row.slot + 1 >= len(g):
            continue
        s = row.sigma
        hi, lo, cl, op = g.high.values, g.low.values, g.close.values, g.open.values
        n = len(g)
        for side in ("short", "long"):
            p = row.p_hi if side == "short" else row.p_lo
            if not np.isfinite(p):
                continue
            running = hi[:row.slot + 1].max() if side == "short" else lo[:row.slot + 1].min()
            entry = op[row.slot + 1]
            stop = running + buffer_sigma * s if side == "short" else running - buffer_sigma * s
            R = (stop - entry) if side == "short" else (entry - stop)
            if R <= 0 or R > max_risk_sigma * s:
                continue
            rec = {"date": row.date, "slot": int(row.slot), "side": side, "p": float(p),
                   "R_pts": float(R), "cost_R": float(COST_SIGMA * s / R),
                   "bars_left": int(n - row.slot - 1)}
            for kt in k_tps:
                tp = entry - kt * R if side == "short" else entry + kt * R
                px, reason = None, "time"
                for j in range(row.slot + 1, n):
                    hit_stop = hi[j] >= stop if side == "short" else lo[j] <= stop
                    hit_tp = lo[j] <= tp if side == "short" else hi[j] >= tp
                    if hit_stop:                       # stop takes priority (conservative)
                        px, reason = stop, "stop"
                        break
                    if hit_tp:
                        px, reason = tp, "tp"
                        break
                if px is None:
                    px = cl[n - 1]
                g_r = ((entry - px) if side == "short" else (px - entry)) / R
                rec[f"g_{kt}"] = float(g_r)
                rec[f"n_{kt}"] = float(g_r - rec["cost_R"])
                rec[f"r_{kt}"] = reason
            recs.append(rec)
    return pd.DataFrame(recs)


def pick(cand: pd.DataFrame, theta: float, kt: float,
         max_per_day: int = MAX_PER_DAY) -> pd.DataFrame:
    """Entries above the threshold, then concurrency control.

    Default: at most ONE trade per instrument per day, taken as the first valid
    signal by (slot, side). Without this cap a single session can open several
    positions on the same side, so one directional bet is counted as many
    independent samples. Positions never overlap in time either, because every
    trade is closed by the session close.
    """
    sub = cand[cand.p >= theta].copy()
    if not len(sub):
        return sub
    sub = sub.sort_values(["date", "slot", "side"])
    if max_per_day is not None and max_per_day > 0:
        sub = sub.groupby("date", sort=True).head(max_per_day)
    sub["net"] = sub[f"n_{kt}"]
    sub["gross"] = sub[f"g_{kt}"]
    sub["reason"] = sub[f"r_{kt}"]
    return sub


def metrics(tr: pd.DataFrame) -> dict:
    """Portfolio metrics, always computed in CHRONOLOGICAL order.

    The out-of-sample trades are concatenated per symbol when built, so an unsorted
    cumsum would produce a drawdown that belongs to no real portfolio.
    """
    if len(tr) == 0:
        return {"n": 0}
    if "date" in tr.columns:
        tr = tr.sort_values(["date", "slot"]).reset_index(drop=True)
    x = tr.net.values
    eq = np.cumsum(x)
    dd = float(np.max(np.maximum.accumulate(eq) - eq))
    w, l = x[x > 0], x[x <= 0]
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean_R": float(x.mean()), "total_R": float(x.sum()),
            "hit": float(np.mean(x > 0)), "std_R": sd,
            "sharpe_trade": float(x.mean() / sd) if sd > 0 else 0.0,
            "max_dd_R": dd, "gross_mean_R": float(tr.gross.mean()),
            "cost_R": float(tr.cost_R.mean()),
            "profit_factor": float(w.sum() / -l.sum()) if len(l) and l.sum() < 0 else np.nan,
            "expectancy_pct_R": float(100 * x.mean())}


def walkforward_select(cand, rt, thetas, k_tps, min_train_days=150, min_trades=40,
                       max_per_day: int = MAX_PER_DAY):
    days = np.array(sorted(rt.date.unique()))
    months = pd.Series(pd.to_datetime(days)).dt.to_period("M").values
    picks, oos = [], []
    for mo in sorted(set(months)):
        past_days = set(np.array(days)[months < mo])
        if len(past_days) < min_train_days:
            continue
        test_days = set(np.array(days)[months == mo])
        best, score = None, -1e9
        for th in thetas:
            for kt in k_tps:
                m = metrics(pick(cand[cand.date.isin(past_days)], th, kt, max_per_day))
                if m["n"] < min_trades:
                    continue
                sc = m["mean_R"] * np.sqrt(m["n"])
                if sc > score:
                    best, score = (th, kt), sc
        if best is None:
            continue
        picks.append({"month": str(mo), "theta": best[0], "k_tp": best[1]})
        t = pick(cand[cand.date.isin(test_days)], best[0], best[1], max_per_day)
        if len(t):
            oos.append(t)
    return (pd.concat(oos) if oos else pd.DataFrame()), picks


def mc_stats(tr: pd.DataFrame, n_boot: int = 5000, block_days: int = 5, seed: int = 0):
    """Bootstrap and permutation on the DAILY P&L series.

    The old version resampled n/5 single days and rescaled the sum by 5, which
    inflates the interval by about sqrt(5) and is not a block bootstrap. This uses
    hodlod.evaluate.block_bootstrap_ci on the daily P&L, then rescales the mean and
    its interval to the horizon by multiplying by the number of days (a total is
    n_days times the mean daily P&L).
    """
    if len(tr) == 0:
        return {}
    tr = tr.sort_values(["date", "slot"])
    daily_pnl = tr.groupby("date").net.sum().sort_index()
    ci = block_bootstrap_ci(daily_pnl.values, block=block_days, n_boot=n_boot, seed=seed)
    n_days = int(len(daily_pnl))
    rng = np.random.default_rng(seed)
    x = tr.net.values
    perm = np.array([float((x * rng.choice([-1, 1], len(x))).sum()) for _ in range(n_boot)])
    return {"boot_mean": float(ci["mean"] * n_days),
            "boot_ci": [float(ci["ci95"][0] * n_days), float(ci["ci95"][1] * n_days)],
            "boot_mean_daily": float(ci["mean"]),
            "p_boot_le_0": float(1.0 - ci["p_positive"]),
            "n_days": n_days, "block_days": block_days,
            "perm_p": float(np.mean(perm >= x.sum())),
            "perm_95": float(np.quantile(perm, 0.95)),
            "boot_dist": (daily_pnl.values[:1500] * n_days).tolist(),
            "perm_dist": perm[:1500].tolist()}


def deflated_sharpe(sr, T, n_trials, sr_std, skew=0.0, kurt=3.0):
    g = 0.5772156649
    N = max(int(n_trials), 2)
    sr0 = float(sr_std) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e)))
    den = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-9))
    return {"sr0": sr0, "dsr": float(norm.cdf((sr - sr0) * np.sqrt(max(T - 1, 1)) / den))}


# ---------------------------------------------------------------- main
THETAS = [0.4, 0.5, 0.6, 0.7, 0.8]
K_TPS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def main():
    res = {"meta": {"cost_sigma": COST_SIGMA, "generated": pd.Timestamp.now().isoformat(),
                    "thetas": THETAS, "k_tps": K_TPS}}

    print("== A. amplitude (6 years daily) ==")
    res["amplitude"] = {}
    for sym in ("ES=F", "NQ=F"):
        a = study_amplitude(sym, 6)
        res["amplitude"][sym] = a
        res.setdefault("freq_amp", {})[sym] = frequency_amplitude(a)
        print(f"  {sym:5s} n={a['n_days']:4d} {a['start']}->{a['end']} | "
              f"E[R+]={a['mean_Rup']:.3f} (theo 0.798) | E[range/sigma]={a['mean_range']:.3f} "
              f"(theo 1.596) | mult={[round(v,2) for v in a['mult']]} | KS={a['ks_Rup']:.3f}")

    print("== C/D. timing + real time (1h, 730 d) ==")
    res["timing"], res["realtime"] = {}, {}
    cands = {}
    for sym in ("SPY", "QQQ", "IWM"):
        t = study_timing(sym)
        res["timing"][sym] = t
        print(f"  {sym}: {t['n_sessions']} sessions | chi2 clock={t['chi2_clock']:.1f} "
              f"(p={t['p_clock_test']:.3f}) | chi2 vol={t['chi2_vol']:.1f} "
              f"(p={t['p_vol_test']:.3f}) | KS PIT clock={t['ks_pit_clock']:.3f} "
              f"vol={t['ks_pit_vol']:.3f}")
        rt = realtime_rows(sym)
        rt.to_pickle(OUT / f"rt_{sym}.pkl")
        b = brier_table(rt)
        res["realtime"][sym] = b
        print(f"     Brier_hi model={b['hi']['brier_model']:.4f} clim={b['hi']['brier_clim']:.4f} "
              f"BSS={b['hi']['bss']:+.4f}")
        bars, _ = sessions_1h(sym)
        cands[sym] = simulate(rt, bars, K_TPS)
        cands[sym].to_pickle(OUT / f"cand_{sym}.pkl")

    print("== in-sample sweep + walk-forward ==")
    res["sweep"], all_oos, picks = [], [], []
    for sym in ("SPY", "QQQ", "IWM"):
        rt = pd.read_pickle(OUT / f"rt_{sym}.pkl")
        for th in THETAS:
            for kt in K_TPS:
                m = metrics(pick(cands[sym], th, kt))
                if m["n"]:
                    res["sweep"].append({"symbol": sym, "theta": th, "k_tp": kt, **m})
        oos, pk = walkforward_select(cands[sym], rt, THETAS, K_TPS)
        picks += [{"symbol": sym, **p} for p in pk]
        if len(oos):
            all_oos.append(oos.assign(symbol=sym))
        print(f"  {sym}: IS best="
              f"{max([s for s in res['sweep'] if s['symbol']==sym], key=lambda z: z['mean_R'])['mean_R']:+.3f} R | "
              f"OOS n={len(oos)}")
    res["picks"] = picks
    if all_oos:
        oos = pd.concat(all_oos)
        oos.to_pickle(OUT / "oos_trades.pkl")
        m = metrics(oos)
        res["oos"] = m
        res["mc"] = mc_stats(oos)
        sweep_r = [s["mean_R"] for s in res["sweep"]]
        res["dsr"] = deflated_sharpe(m["sharpe_trade"], m["n"],
                                     len(THETAS) * len(K_TPS) * 3, float(np.std(sweep_r)))
        res["freq_signal"] = {
            "n_days_total": int(len(set(oos.date))),
            "n_trades": int(len(oos)),
            "pct_days_with_signal": float(len(set(oos.date)) / len(set(pd.read_pickle(OUT / 'rt_SPY.pkl').date))),
            "tp_rate": float(np.mean(oos.reason == "tp")),
            "sl_rate": float(np.mean(oos.reason == "stop")),
            "time_rate": float(np.mean(oos.reason == "time")),
        }
        print(f"  OOS n={m['n']} hit={m['hit']:.3f} mean_R={m['mean_R']:+.4f} "
              f"total_R={m['total_R']:+.1f} DD={m['max_dd_R']:.1f} "
              f"PF={m['profit_factor']:.2f}")
        print(f"  MC CI95={[round(x,1) for x in res['mc']['boot_ci']]} "
              f"p(<=0)={res['mc']['p_boot_le_0']:.3f} perm_p={res['mc']['perm_p']:.3f}")
        print(f"  DSR={res['dsr']['dsr']:.3f} (sr0={res['dsr']['sr0']:.3f})")
    (OUT / "results.json").write_text(json.dumps(res, default=float))
    print("\n-> research/out/results.json")


if __name__ == "__main__":
    main()
