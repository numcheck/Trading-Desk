# Estimating and trading the HOD / LOD of the RTH session — ES & NQ

Working document. Code: `hodlod/`. Self-tests: `python -m hodlod.selftest` (21/21 pass).

> **Update — empirical results (see `research/out/HOD_LOD_ES_NQ.pdf`).**
> The protocol of sections 8 to 10 was run on real free data (Yahoo Finance):
> 1,383 ES & NQ sessions daily (2021–2026), 491 sessions × 3 instruments hourly
> (2024–2026). Measured results, not simulated:
>
> | What is tested | Result | Verdict |
> |---|---|---|
> | Amplitude `E[range]/σ` (theory 1.596) | **1.80 – 1.82** | diffusion understates by ~13% |
> | Amplitude multiplier q25→q95 | **1.06 – 1.12**, flat | correction by a constant `κ ≈ 1.09` |
> | Band `[q25,q75]` of `R+` (nominal 50%) | **50.3%** | calibrated |
> | Arcsine on the raw clock | **rejected** (χ² 34.5 / 53.5 / 108.8) | the null is false |
> | Arcsine in volatility time | χ² reduced by 40%, KS ÷ 2.6 | better, but **still rejected** |
> | `P(running HOD = final)` vs hourly climatology | **BSS +0.33 to +0.39** | the formula genuinely predicts |
> | Walk-forward backtest, 521 trades | gross **+0.0496 R**, cost **0.0302 R**, net **+0.0194 R** | positive, **not significant** |
> | DSR (90 registered trials) | **0.012** | nothing proven |
>
> Consequence for section 11: the estimate "25–40% chance of a real edge" was the right
> range, and the hard point is not the signal but the **cost per unit of risk**: the
> round trip would have to fall below 1% of σ for the same signal to become profitable.

---

## 0. Explicit assumptions (in lieu of questions)

| # | Assumption |
|---|---|
| H1 | Universe: ES and NQ, front-month continuous adjusted by difference, 1-minute bars, RTH 09:30–16:00 ET. `O` = first trade ≥ 09:30, `H`/`L` = max/min of high/low over [09:30, 16:00], `C` = last trade ≤ 16:00. |
| H2 | ≥ 2,500 sessions (≈ 10 years) of 1-minute data. Below ~1,000 sessions, any model with more than 3–4 parameters is undecidable, whatever the p-value displayed. |
| H3 | `σ_day` = standard deviation of the full RTH session. It is only known ex post: the whole problem is to **forecast** it, then to **update** it, and to work in units of σ. |
| H4 | No market data were available in this environment. The **empirical frequencies** (level rejection rates, real PIT, edge) are therefore **not** supplied: the estimator, its confidence interval and the placebo control are. Inventing those numbers would be the most serious possible fault here. |
| H5 | Costs: ES 0.25 pt = $12.50/tick, $50/pt; NQ 0.25 pt = $5/tick, $20/pt. Round trip at market ≈ 1.5–2 ticks all-in. Typical RTH range: ES 40–60 pts (160–240 ticks) → cost ≈ 0.8–1.2% of the range; NQ 250–400 pts (1,000–1,600 ticks) → ≈ 0.15%. **In relative ticks, NQ is more tradable; ES has a better signal-to-noise ratio per point.** |
| H6 | Intraday drift is not assumed zero. Every formula is given with and without drift. |
| H7 | `HOD_final` is only known at 16:00. The label "the running max is the final one" is therefore an end-of-day label: usable only with a one-day lag in training. |
| H8 | Single session, no Globex in the features (except the overnight range, known at 09:29). |

---

## 1. The problem, stated correctly

The HOD is not a level, it is an **argmax**. Three consequences drive the whole design:

1. **There is no "HOD level" to predict.** There is a distribution of `(H−O)/σ` and a
   distribution of the time `T_H`. Looking for an exact price is a specification error.
2. **Two distinct questions.** Ex ante: "where will the final max be?" → a distribution.
   In real time: "is the running max the final one?" → a survival probability. Only the
   second is actionable during the session.
3. **The event is absorbing.** `1{HOD_t = HOD_final}` equals 1 as soon as the max is in,
   so its probability climbs mechanically towards 1 late in the session. This is a
   **survival** problem, not a classification problem. Any AUC-style metric will be
   flattering and misleading (cf. §8).

The only genuinely exploitable information is therefore:
- the **amplitude** (how far the session can travel),
- the **asymmetry** (which side extends),
- the **timing** (when the max is probably in).

Amplitude is highly predictable (it is volatility). Asymmetry is barely predictable.
That is where the edge lives, and that is where it is thin.

### Most useful calibration identity

For a driftless BM on [0,T], `E[max] = σ√T·√(2/π) = 0.7979·σ√T` and
`E[range] = √(8/π)·σ√T = 1.5958·σ√T`. Hence:

```
σ_day  ≈  range_RTH / 1.596
```

An ES ranging 50 points has `σ_day ≈ 31 points`. This identity is a permanent coherence
check across your three sources of σ (range, 1-minute RV, HAR).

---

## 2. Ranking by real relevance for locating the HOD/LOD

Score 1 to 5. "Real role" = what the model actually does, not what we would like it to do.

| # | Model | What it estimates exactly | Score | Real role |
|---|---|---|---|---|
| 1 | **Maximum survival / first passage** (§6) | `P(sup_{[t,T]} X < M_t | F_t)` | **5** | The only object that answers the real-time question. It is the trigger. |
| 2 | **CARR** (Chou 2005) | `λ_t = E[H_t−L_t | F_{t−1}]`, full conditional distribution via `ε` | **4** | Conditional scale of the amplitude. The denominator of everything else. |
| 3 | **HAR-RV** (Corsi 2009) | `E[RV_{t+1} | RV_t, RV̄_5, RV̄_22]` | **4** | Indispensable input (`σ_day`). Locates **nothing**. |
| 4 | **Quantile regression / QRF / pinball boosting** | `q_α((H−O)/σ | X)`, `q_α((O−L)/σ | X)` | **4** | The pre-open map. The only model that gives a region. |
| 5 | **Arcsine law in volatility time** | `P(T_H ≤ t) = (2/π)arcsin√F(t)` | **3.5** | Free timing prior, zero parameters. Benchmark to beat. |
| 6 | **Two-sided barriers** (range) | `P(−a < min, max < b)` | **3** | Probability that the rest of the session stays in a corridor → range fade. |
| 7 | **EVT / GPD on normalised amplitude** | tail law of `R±` beyond a threshold | **3** | Stop placement, tail size, not localization. |
| 8 | **Confluence levels + placebo** (§5) | rejection frequency conditional on distance | **3** | Trigger filter. **Not validated without the placebo control.** |
| 9 | **Range estimators** (Parkinson, GK, RS, YZ) | the day's `σ²`, from OHLC | **2.5** | **Measurement** instruments. They improve the RV label, not the forecast. |
| 10 | **HEAVY / Realized GARCH** | conditional variance + realized measure | **2.5** | Alternative to HAR for `σ_day`. Same role. |
| 11 | **CAViaR** (Engle & Manganelli 2004) | `q_θ(r_t | F_{t−1})` — a **return** quantile | **1.5** | **Something else: VaR.** Useful for sizing the stop, useless for locating H/L. |

### What serves another purpose (to state clearly)

- **CAViaR** → VaR / Expected Shortfall. The quantile of a return is not the max of a
  path. Confusing the two is the most common error in this type of project.
- **EVT** → return tails, extreme VaR, stress testing. Reusable for amplitude **only**
  after normalisation by σ (otherwise you measure the seasonality of volatility, not the
  tail).
- **Yang-Zhang** → variance estimation for option pricing and volatility forecasting. The
  most efficient of the four, and the only one that survives gaps, but it needs a window
  (n ≥ 2) and says nothing about H/L.
- **HAR-RV / HEAVY** → volatility forecasting. Excellent at what it does (OOS R² 0.4–0.6
  in logs), with no directional content.
- **Parkinson / GK / RS** → measurement. Their interest here is indirect: a better `RV_t`
  improves the HAR, which improves `σ_day`, which improves every probability.

> **Hard point:** none of these models "locates" the HOD in the sense of "here is the
> price". They produce a region and a probability. A system claiming to give an exact
> price has no statistical basis.

---

## 3. A — Amplitude: `(H−O)/σ` and `(O−L)/σ`

### A.0 The exact benchmark, and it is free

Under a driftless BM, `H−O` has the law of `|N(0, σ²T)|` (a direct consequence of the
reflection principle):

```
P(H−O ≤ x) = 2Φ(x / S) − 1 ,     S = σ√T
```

Hence, with no parameter to estimate:

| Quantity | Value |
|---|---|
| `E[(H−O)/S]` | `√(2/π) = 0.7979` |
| median | `Φ⁻¹(0.75) = 0.6745` |
| `q_90` | `Φ⁻¹(0.95) = 1.6449` |
| `q_95` | `Φ⁻¹(0.975) = 1.9600` |
| `E[(H−O)+(O−L)]/S` | `1.5958` (= `√(8/π)`, the expected range) |

Any systematic deviation from this law is a **tail / jump / drift premium** to be
measured, not assumed. The most informative diagnostic is the **amplitude multiplier**:

```python
# hodlod/amplitude.py
amplitude_multiplier(R_plus_standardised)  # -> {q10: m10, ..., q95: m95}
```

Reading:
- **flat ratio > 1** → a simple scale error on σ (your σ is too small);
- **ratio increasing with α** → fat tails (jumps, stochastic vol);
- **ratio > 1 at both ends** → monitoring bounds + jumps.

**Measured result (Monte-Carlo, 8,000 paths, 1,200 monitoring points):** the empirical
CDF of the max is 0.6933 against 0.6827 in theory at `x = 1σ`. In other words, **discrete
monitoring understates the amplitude by about 1%** (extremes between two bars are
invisible). Practical corollary: the continuous formula is slightly **optimistic**;
calibrate a monitoring factor `κ` such that `S_eff = κ·σ√T` with `κ ≈ 0.99` at 1-minute
resolution, `κ → 1` at tick resolution.

### A.1–A.4 Range estimators: formulas, assumptions, where they break

| Estimator | Formula | Efficiency vs C2C | Assumptions | Where it breaks |
|---|---|---|---|---|
| Parkinson (1980) | `σ²_P = (ln(H/L))² / (4 ln2)` | 5.2× | driftless BM, **no gap**, continuous monitoring | opening gap → downward bias; drift → upward bias |
| Garman-Klass (1980) | `σ²_GK = ½(ln H/L)² − (2ln2−1)(ln C/O)²` | 7.4× | same + no gap | same |
| Rogers-Satchell (1991) | `σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` | ~8× | BM **with drift** allowed, no gap | opening gap |
| Yang-Zhang (2000) | `σ²_YZ = σ²_on + k·σ²_oc + (1−k)·σ²_RS`, `k = 0.34/(1.34+(n+1)/(n−1))` | ~14× | drift + gaps, window n ≥ 2 | little; the most robust, but it is a window average, not a daily estimator |

**Central message: these are measurement instruments, not forecasts.** They answer "what
was today's volatility?", not "what will tomorrow's range be?". Their correct use here:

1. build a clean `RV_t` (1-minute bars) and a jump-robust `RV_t`
   (`BV = (π/2)Σ|r_t||r_{t−1}|`, Barndorff-Nielsen & Shephard 2006);
2. provide regressors to the HAR (`log RP_prev`, `log GK_prev`, `log RS_prev`);
3. check the coherence `σ_day ≈ range/1.596`.

### A.5 EVT / GPD on the normalised amplitude

**What it estimates:** the tail law of `R⁺ = (H−O)/σ_day` beyond a threshold `u`.

```
u = q_F(R⁺) ;  Y = R⁺ − u | R⁺ > u  ~  GPD(ξ, β)
P(R⁺ > u+y) = (1−F_u)·(1 + ξy/β)^(−1/ξ)
q_α(R⁺) = u + (β/ξ)[ ((1−α)/(1−F_u))^(−ξ) − 1 ]
```

**Data:** 2,500+ sessions of `R⁺` **normalised by the forecast σ_day** (never by realised
σ: that would be look-ahead).
**Calibration:** threshold at the 90th percentile; check the **stability of ξ** over a grid
of thresholds (stability plot); `scipy.stats.genpareto.fit(exc, floc=0)`.
**Assumptions:** regularly varying tail, approximately iid excesses.
**Where it breaks:** (i) the threshold choice drives ξ; (ii) excesses are **clustered**
(crisis days) → decluster or use a conditional model; (iii) non-stationarity: without
normalisation you measure the seasonality of volatility, not the tail; (iv) at 2,500
observations only ~250 excesses remain, and the CI on ξ is about `±(1+ξ)/√250 ≈ ±0.08` —
wide.

**Validation:** coverage of `q_99` and `q_995` (must be 99% and 99.5% ± binomial CI), and
pinball loss above the threshold.
**Measured on a Pareto(ξ=1/3) simulation, n=50,000:** ξ estimated 0.374 (true 0.333),
`q_99` at +0.7% — the estimator is correct but ξ is noisy.

### A.6 HAR-RV

```
RV_{t+1} = c + β_d·RV_t + β_w·RV̄_{t−4..t} + β_m·RV̄_{t−21..t} + ε
```

In practice we estimate on `log RV` (stabilises heteroscedasticity) and transform back
with the smearing correction `exp(σ̂²_ε/2)`.

**Data:** `RV_t = Σ r²_{1min}` (or `BV` for the continuous part). The `HARQ` version
(Bollerslev, Patton & Quaedvlieg 2016) adds `RQ_t^{1/2}` to correct the measurement-error
bias — a real gain on high-frequency data.
**Calibration:** OLS; HAC errors. For inference on `log RV`, OLS is enough.
**Assumptions:** hierarchical structure (traders at 1-day/1-week/1-month horizons), stable
coefficients, RV without microstructure noise.
**Where it breaks:** regime breaks (March 2020), contract roll (an RV computed on an
unadjusted price explodes on the roll day), half sessions, and 1-minute bars on NQ where
microstructure noise is stronger.
**Validation:** QLIKE and MSE on out-of-sample RV, plus a Mincer-Zarnowitz regression
`RV = a + b·RV̂` (test `b=1`).

### A.7 CARR — and the answer to your question about the founding paper

**You are right.** The founding paper is:

> Chou, R. Y. (2005). *Forecasting Financial Volatilities with Extreme Values: The
> Conditional Autoregressive Range (CARR) Model.* **Journal of Money, Credit and Banking,
> 37(3), 561–582.** DOI 10.1353/mcb.2005.0027.

What **Chen, Gerlach & Lin (2008)** actually is:

> Chen, C. W. S., Gerlach, R., & Lin, E. M. H. (2008). *Volatility forecasting using
> threshold heteroskedastic models of the intra-day range.* **Computational Statistics &
> Data Analysis, 52(6), 2990–3010.**

It is a **threshold extension** of CARR (sign/size asymmetry, Bayesian MCMC estimation),
not the founding paper. Attributing CARR to Chen et al. (2008) is a citation error.

**Model (Chou 2005):**

```
R_t = λ_t · ε_t ,        ε_t iid, E[ε] = 1, ε ⊥ F_{t−1}
λ_t = ω + Σ_{i=1..p} α_i R_{t−i} + Σ_{j=1..q} β_j λ_{t−j}
```

`CARR-X` extension (same paper): `+ Σ γ_k x_{k,t−1}` for exogenous variables (VIX,
overnight RV).

**What it estimates exactly:** `λ_t = E[R_t | F_{t−1}]`, the **expected amplitude of the
range**, plus the full conditional law via `ε`. It is **not** a variance forecast, even
though `λ_t` is a monotone proxy for one.

**Conditional distribution:** if `ε ~ Exp(1)`, then `q_α(R_t) = −λ_t ln(1−α)`. Immediately
usable as an amplitude map.

**Calibration:** quasi-ML (the exponential likelihood remains consistent if `ε` is not
exponential, as long as `E[ε]=1`).

```python
from hodlod.vol import CARR
m = CARR().fit(R)                      # R = daily range in points
lam_next = m.predict_next(R)           # E[tomorrow's range | history]
q80 = CARR.quantile(lam_next, 0.80)    # q80 of tomorrow's range
m.persistence, m.se_                   # alpha+beta, standard errors (BHHH)
```

**Assumptions:** linearity in `R`, no threshold, `ε` of mean 1 and iid.
**Where it breaks:** (i) event days — the range explodes without `λ_t` anticipating it;
(ii) `β → 1` ⇒ under-reaction to breaks; (iii) the asymmetry documented by Chen et al.
(2008): a down shock does not affect the range like an up shock; (iv) the range is bounded
by session liquidity (half sessions).
**Validation:** QLIKE on the range (`QLIKE = R/λ − ln(R/λ) − 1`), and pinball loss of the
quantiles `λ_t·(−ln(1−α))`.

**Numerical check:** on 20,000 simulated observations with `(ω,α,β) = (0.25, 0.10, 0.85)`,
the QMLE recovers `(0.0999, 0.8583)`, persistence 0.958 against 0.95. The implementation
is correct.

### A.8 The amplitude model everyone forgets: the **remaining** part

The range already travelled consumes the budget. At 11:00, the expected `H−O`
conditional on a range that is already large is **smaller**, not larger. The right object
is therefore:

```
R⁺_remaining = (H_final − H_t) / σ_remaining ,    σ_remaining = σ_day·√(1 − F(t))
```

which is **exactly** the same survival function as in §6. There are not two models:
amplitude and real time are the same mathematical object seen from two sides. This is the
most important architectural point in the document.

---

## 4. B — Localization

### B.1 "Which side first" probability (with drift)

Under a BM with total drift `M` over the horizon and total standard deviation `S`,
starting from the open:

```
P(touch +b before −a) = (1 − e^(−2Ma/S²)) / (1 − e^(−2M(a+b)/S²))
```

At `M = 0`: `a/(a+b)`. This is the only formula in the document that contains a
**direction**, and it is entirely determined by the drift. Any asymmetry you can trade
comes from there, or does not exist.

### B.2 Quantile regression (Koenker & Bassett 1978)

```
q_α(Y | X) = argmin_θ  Σ ρ_α(Y_i − X_i'θ),   ρ_α(u) = u(α − 1{u<0})
```

**Targets:** `R⁺ = (H−O)/σ̂` and `R⁻ = (O−L)/σ̂`. Useful quantiles: 0.25 / 0.50 / 0.75 / 0.90.
**Features (all known before 09:30):** `log RV_1,5,22`, `log RP_prev`, `range_prev/σ`,
`gap/σ`, `range_overnight/σ`, `VIX`, `ΔVIX`, day of week, `is_event` (FOMC/CPI/NFP,
calendar known ex ante), position of the previous close inside its range.
**Calibration:** `statsmodels.QuantReg`. Standard errors: block bootstrap (days), not
asymptotic SEs.
**Assumptions:** linearity, non-crossing quantiles, stationarity.
**Where it breaks:** quantile crossings (frequent at 0.90), uncaught heteroscedasticity,
and above all **few effective observations**: 2,500 days, 12 features, 4 quantiles →
estimation variance quickly dominates.
**Validation:** out-of-sample pinball loss, **coverage** of `[q05, q95]` (must be 90%), and
`CRPS = 2∫₀¹ QL_τ dτ`.
**Measured on simulation:** coverage 0.906 for a nominal `q_0.9` of 0.90.

### B.3 Quantile Regression Forests and pinball boosting

- **QRF** (Meinshausen 2006): random forests + conditional quantile estimation by weighting
  observations. Consistent, non-parametric, handles interactions. Risk: slow, and extreme
  quantiles are poorly estimated (leaves hold few observations).
- **Pinball boosting:** `sklearn.GradientBoostingRegressor(loss="quantile", alpha=α)`.
  Faster, better on interactions. Set `max_depth ≤ 3`, `min_samples_leaf ≥ 40`,
  `subsample = 0.8`.
- One model **per quantile** independently ⇒ crossings. Check with
  `amplitude.crossing_check` and, if the crossing rate exceeds 5%, reorder the predictions
  (sort by quantile).

**The real gain of ML here is not average accuracy: it is regime dependence** (event day ×
high vol × gap). A linear regression does not capture `is_event × vol`.

### B.4 Asymmetry classifier — the only directional signal in the framework

Target: `1{R⁺ > R⁻}`, i.e. "does the upper branch dominate?".

Under a driftless BM: **exactly 0.5**. There is therefore nothing to find, except drift,
autocorrelation or microstructure.

The expectation-setting fact: Gao, Han, Li & Zhou (2018) document a robust **intraday
momentum** (the first half-hour's return predicts the last one) with an **out-of-sample
R² of 1.4% to 2.0%**, and up to 3.3% on high-volatility days. That is an effect published
in the *Journal of Financial Economics*, replicated across 16 markets — and its R² is 2%.
That is the order of magnitude of **everything** you can hope for on asymmetry.

Corollary: an asymmetry model with an R² of 15% is a bug, not a discovery.

### B.5 Confluence levels: the only method that does not lie to itself

Levels: `PDH`, `PDL`, `PDC`, `PD VWAP`, `ONH`, `ONL`, `IBH`, `IBL` (known at 10:00),
current `VWAP`, `POC`/`VAH`/`VAL`, round numbers, weekly and monthly high/low.

**The trap:** "the PDH is touched 70% of the time" means nothing. A level close to the
price is touched more often, mechanically. Three mandatory steps:

1. **Condition on distance** `|level − O|/σ_day`.
2. **Matched-distance placebo control:** draw a fake level at the same distance (empirical
   distribution of real distances) and compute the same statistic. If the difference is not
   significant, the level adds nothing.
3. **Symmetric definitions:** touch = wick inside `[L−tol, L+tol]`; break = close beyond
   `L+tol`; rejection = touch without break **and** a return beyond `3·tol` on the origin
   side.

```python
from hodlod.levels import confluence_table
confluence_table(days, "pdh", tol_sigma=0.05, placebo=True)
# -> {'n':..., 'reject_rate':..., 'placebo_reject_rate':..., 'delta':..., 'delta_ci95':[...], 'significant':bool}
```

**The only interpretable number is `delta` and its CI.** The rest is noise dressed up as
statistics.

**Where it breaks:** levels are correlated with each other (PDH and ONH often coincide), so
a per-level test overstates global significance; a joint test is needed (Bonferroni or
Hansen's 2005 SPA), with an effective `n` that accounts for regime clustering.

---

## 5. C — Timing

### C.1 The reference law, free and testable

Third arcsine law (Lévy 1939/1948): for a driftless BM on [0,1],

```
P(T_H ≤ t) = (2/π)·arcsin(√t)          density 1/(π√(t(1−t)))
```

Key figures:

| `t` | `P(T_H ≤ t)` |
|---|---|
| 0.10 | **0.205** |
| 0.25 | 0.333 |
| 0.50 | 0.500 |
| 0.75 | 0.667 |
| 0.90 | 0.795 |

**41% of sessions place their high in the extreme 10% of the day** (first or last 48
minutes). This is not an empirical regularity of intraday volatility: it is a property of
**any** continuous random walk. It is therefore a prior, not an anomaly.

The same holds for `T_L`, and `P(T_H < T_L) = 1/2` without drift.

### C.2 Empirical validation on ES

> Mok, D. M. Y., Lam, K., & Li, W. (2000). *Using Daily High/Low Time to Test for Intraday
> Random Walk in Two Index Futures Markets.* **Review of Quantitative Finance and
> Accounting, 14, 381–397.**

They derive the theoretical law of the time of the high/low under a random walk (at
uniform **or** variable trading speed), and conclude that **for S&P 500 futures the random
walk hypothesis is not rejected**, whereas it is for the Hang Seng (extremes too often in
the first 15 minutes). This is the most directly relevant reference for your universe, and
it is rather unfavourable: on ES, the timing of the high/low is already well described by
the null model.

### C.3 Volatility time: `F(t)` rather than `t`

The intraday seasonal profile (Andersen & Bollerslev 1998) is not a cosmetic correction, it
is a **change of clock**. Define

```
s(t) = normalised variance profile (Σ s = 1),   F(t) = ∫₀ᵗ s(u)du
```

Then the timing prior becomes `P(T_H ≤ t) = (2/π)arcsin√F(t)`, and
`σ_remaining(t) = σ_day·√(1−F(t))`.

**Measured (Monte-Carlo, pronounced U-shaped profile, 4,000 paths, 1,000 points):** on the
raw clock, the KS distance to the reference law is **0.101**; in volatility time, **0.031**.
A factor-3.3 gain — real but **not perfect**: the time change is an approximation, and the
profile is estimated, not known. Never present this transformation as exact.

**Diagnostic trap:** under H0, `F(T_H)` follows an **arcsine** law, not a uniform one. The
correct PIT is `u = ArcsinCDF(F(t_max))`. Forgetting the second transformation wrongly
rejects H0 — the most frequent error in this diagnostic (and one I made in the first
version of the code; the tests caught it).

### C.4 Hazard and competing risks

Non-parametric version, directly comparable to the diffusion model:

```
h_k = P(first crossing in bin k | not yet crossed)
S_k = Π_{j≤k}(1−h_j)
```

`S_k` is the probability that the running max is still the final one at bin `k`. It is the
target of a classifier and the control for the analytic model.

`timing.competing_risk_table` additionally gives `P(H and L both already set)`, which
defines the **range-fade regime**: a structurally different strategy from extreme
reversal, and probably more robust.

### C.5 Where timing breaks

1. **Fixed-time announcements:** 10:00 (ISM, confidence), 14:00 (FOMC), 14:30 (press
   conference), 8:30 (CPI/NFP, before the open). The hazard is not smooth: it has
   deterministic spikes.
2. **Auctions:** the open (09:30) and the close (15:50–16:00) have their own microstructure.
   The "last-minute high" is often an auction artefact.
3. **Half sessions** (13:00 ET): `F(t)` changes shape. Exclude them or treat them separately.
4. **Intraday momentum:** trend days make a new high in the last 30 minutes more often than
   the arcsine law predicts. This is precisely the Gao et al. (2018) effect.

---

## 6. D — Real time: "is the running HOD the final one?"

### D.1 The formula, its derivation, and its five failure modes

Let `M_t` be the running max, `X_t` the price, `d = M_t − X_t ≥ 0`, and `S = σ·√τ` the
standard deviation of the price over the remaining horizon. For a driftless BM starting at
`X_t`:

```
P(sup_{[t,T]} X < M_t | F_t) = P(sup_{u≤τ} (X_t + σW_u) < X_t + d)
                            = P(sup W < d/σ√τ)
                            = 2Φ(d/S) − 1
```

This is exactly the formula in the brief. It is **exact** under its assumptions, and it has
two useful limits: `d→0 ⇒ p→0`, `d/S→∞ ⇒ p→1` (slowly, as `1 − 2φ(z)/z`).

**With drift** (absorbing-barrier formula):

```
P(max < a) = Φ((a−M)/S) − e^(2Ma/S²)·Φ((−a−M)/S)          M = total drift over the horizon
```

Checked: `M→0` recovers `2Φ(a/S)−1` (measured error 3e-13).

**The five violations, and their direction:**

| # | Violation | Effect on `p` | Order of magnitude |
|---|---|---|---|
| 1 | **Drift** | depends on sign; an upward drift **lowers** `p_H` | `M/S ≈ 0.03` for the average equity drift (negligible); `0.5–1` on trend days (decisive) |
| 2 | **Fat tails / jumps** | `p` **overstated** (a jump can cross a distant barrier) | the dominant term on news days |
| 3 | **Stochastic volatility** | same direction (mixing σ fattens the tails) → `p` overstated | `p_final_mixture(vol_of_vol=0.35)` |
| 4 | **Discrete monitoring** | `p` **understated** (fewer opportunities to cross) | **+0.015 measured** at 1,000 points |
| 5 | **Autocorrelation / momentum** | `ρ>0` ⇒ `p` understated | the Gao et al. effect |

Effects 2/3 and 4 go in **opposite** directions and partly offset. That is why the formula
"works better than expected" in practice — and it is a coincidence, not a validation.
Never use it without recalibration.

**Monte-Carlo check (5,000 paths, 2,000 points):**

| Test | Formula | Empirical |
|---|---|---|
| no drift | 0.4906 | 0.5060 |
| drift +0.4 | 0.3628 | 0.3690 |
| drift −0.4 | 0.6221 | 0.6360 |

The systematic gap (+0.010 to +0.015) is discrete monitoring, exactly as predicted. The
conditional structure is correct.

### D.2 The layer that makes the object usable

```
p_final(t) = g( p_diffusion(d_t, S_t),  features_t )
```

where `g` is an **isotonic calibration** estimated walk-forward, and the features are:
`d/S`, `F(t)`, `σ̂` (level), `vol_ratio = realized RV / expected RV`, `range_so_far/σ`,
`asym_amp`, `dist_vwap/σ`, `log volume`, trend slope, distance to confluence levels,
proximity of an announcement, day type.

**The AUC trap.** The base rate of `1{HOD_t = HOD_final}` climbs towards 1 late in the
session. A classifier that learns "it is 15:55" gets an AUC of 0.97 with no information.
Therefore:
- evaluate the Brier score **at fixed times** (10:00, 11:00, 12:30, 14:00, 15:00),
- compare against the **analytic model**, not against climatology,
- report the AUC but never optimise it.

Mandatory formula:

```
BSS = 1 − BS(model) / BS(diffusion)
```

A `BSS > 0` means "better than the theory". That is the only result worth anything.

### D.3 Two-sided version (range fade)

`p_range_within(a, b, S)` gives the probability that the rest of the session stays inside
`[X−a, X+b]`. Two-barrier image series:

```
P(−a < min, max < b) = Σ_k [Φ((b+2kL)/S) − Φ((−a+2kL)/S)] − Σ_k [Φ((b+2a+2kL)/S) − Φ((a+2kL)/S)]
```

with `L = a+b`. **The second sum is essential:** omitting it gives 0.68 instead of 0.37 for
`a=b=S` (a bug found by the tests). Validated by Monte-Carlo at 0.371 against 0.380 (the
gap is discrete monitoring).

---

## 7. References — verified one by one

### Peer-reviewed papers (verified)

| Reference | Journal / vol / pages | Check |
|---|---|---|
| Chou, R. Y. (2005). Forecasting Financial Volatilities with Extreme Values: The Conditional Autoregressive Range (CARR) Model | *J. Money, Credit and Banking* 37(3), 561–582 | ✅ **CARR founding paper** |
| Chen, C. W. S., Gerlach, R., Lin, E. M. H. (2008). Volatility forecasting using threshold heteroskedastic models of the intra-day range | *Computational Statistics & Data Analysis* 52(6), 2990–3010 | ✅ threshold extension, **not** the founding paper |
| Parkinson, M. (1980). The Extreme Value Method for Estimating the Variance of the Rate of Return | *Journal of Business* 53(1), 61–65 | ✅ |
| Garman, M. B., Klass, M. J. (1980). On the Estimation of Security Price Volatilities from Historical Data | *Journal of Business* 53(1), 67–78 | ✅ |
| Rogers, L. C. G., Satchell, S. E. (1991). Estimating Variance from High, Low and Closing Prices | *Annals of Applied Probability* 1(4), 504–512 | ✅ |
| Yang, D., Zhang, Q. (2000). Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices | *Journal of Business* 73(3), 477–492 | ✅ |
| Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility | *J. Financial Econometrics* 7(2), 174–196 | ✅ |
| Bollerslev, T., Patton, A. J., Quaedvlieg, R. (2016). Exploiting the errors: A simple approach for improved volatility forecasting (HARQ) | *Journal of Econometrics* 192(1), 1–18 | ✅ |
| Engle, R. F., Manganelli, S. (2004). CAViaR: Conditional Autoregressive Value at Risk by Regression Quantiles | *J. Business & Economic Statistics* 22(4), 367–381 | ✅ (NBER WP 7341, 1999) |
| Koenker, R., Bassett, G. (1978). Regression Quantiles | *Econometrica* 46(1), 33–50 | ✅ |
| Meinshausen, N. (2006). Quantile Regression Forests | *JMLR* 7, 983–999 | ✅ |
| Shephard, N., Sheppard, K. (2010). Realising the future: forecasting with high-frequency-based volatility (HEAVY) models | *J. Applied Econometrics* 25(2), 197–231 | ✅ |
| Andersen, T. G., Bollerslev, T. (1998). Deutsche Mark–Dollar Volatility: Intraday Activity Patterns, Macroeconomic Announcements, and Longer Run Dependencies | *Journal of Finance* 53(1), 219–265 | ✅ |
| Mok, D. M. Y., Lam, K., Li, W. (2000). Using Daily High/Low Time to Test for Intraday Random Walk in Two Index Futures Markets | *Review of Quantitative Finance and Accounting* 14, 381–397 | ✅ **directly on S&P futures** |
| Gao, L., Han, Y., Li, S. Z., Zhou, G. (2018). Market intraday momentum | *J. Financial Economics* 129(2), 394–414 | ✅ |
| Holmberg, U., Lönnbark, C., Lundström, C. (2013). Assessing the profitability of intraday opening range breakout strategies | *Finance Research Letters* 10(1), 27–33 | ✅ |
| Bailey, D. H., López de Prado, M. (2014). The Deflated Sharpe Ratio | *J. Portfolio Management* 40(5), 94–107 | ✅ |
| Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. (2017). The Probability of Backtest Overfitting | *J. Computational Finance* 20(4), 39–69 | ✅ |
| Gneiting, T., Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation | *JASA* 102(477), 359–378 | ✅ |
| Brier, G. W. (1950). Verification of forecasts expressed in terms of probability | *Monthly Weather Review* 78(1), 1–3 | ✅ |

### Theory (textbooks)

- Lévy, P. (1939, 1948) — arcsine laws. See Karatzas & Shreve (1991), *Brownian Motion and
  Stochastic Calculus*, problem 8.16: `P(θ_t ≤ s) = (2/π)arcsin√(s/t)`.
- Feller, W. (1971), *An Introduction to Probability Theory and Its Applications*, vol. II —
  barriers, range law.
- Borodin, A. N., Salminen, P. (2002), *Handbook of Brownian Motion: Facts and Formulae* —
  two-barrier image series.
- López de Prado, M. (2018), *Advances in Financial Machine Learning*, ch. 7 — purging and
  embargo.

### Recent work that is **not** peer-reviewed (treat it as such)

These three count for the honest edge estimate, not as proof:

- Fetna, M. (2026). *Opening-Range Breakout Does Not Survive Trading Costs: A
  Pre-Registered 225-Cell Study on Sixteen Years of Futures Data.* SSRN 7428398. → 0 cells
  out of 225 pass the threshold; +$1.57M gross, −$6.49M net at $25 round trip; **median
  gross edge −0.01 tick**.
- Chuk, J. (2026). *Regime-Dependent Opening Range Breakout Returns in Nasdaq-100 Futures:
  A Pre-Registered Study.* Working paper. → 1 survivor out of 144 pre-registered tests on NQ.
- Subhani, S. (2026). *Opening-Range Breakouts on Event-Free Fridays.* SSRN 7589980. → DSR
  0.78 (QQQ) / 0.42 (SPY) after deflation over 57 configurations.

**What these papers really say:** on your exact universe (ES/NQ, 1-minute, 2010–2026), the
"opening range breakout" family produces a gross edge **below the transaction cost**. That
is the base rate to keep in mind for any intraday extreme strategy.

---

## 8. Validation protocol

### 8.1 Splitting

```
purged_walk_forward(n_days, train=1500, test=125, embargo=5, label_horizon=0)
```

The `HOD_final` label is known at 16:00 ⇒ to predict an event of day J, we train on ≤ J−1.
No exceptions.

### 8.2 Anti-look-ahead checklist (specific to this problem)

- [ ] `σ_day` is a **forecast**, never `RV_J`. Test: correlate `σ̂_J` with `RV_J`; if the
  correlation exceeds ~0.6 out of sample, there is a leak.
- [ ] The `F(t)` profile is estimated on a rolling window **that does not contain J** (or on
  a frozen historical average).
- [ ] `VWAP_t` uses only bars `≤ t`.
- [ ] `IBH/IBL` are `NaN` before 10:00.
- [ ] `is_event` comes from a calendar **known ex ante** (FOMC/CPI dates are published).
- [ ] The contract roll is decided on volume/OI **with a one-day lag**.
- [ ] Time zone: 09:30 ET is not a fixed UTC hour. Convert to `America/New_York`, not
  UTC−4/−5.
- [ ] Half sessions excluded or handled separately.
- [ ] Days whose high is made in the last minute are excluded from the evaluation of
  `1{HOD final}` (degenerate label).
- [ ] `leakage_audit(feature_names)` returns nothing.

### 8.3 Metrics, per sub-problem

| Sub-problem | Primary metric | Secondary metric | Acceptance threshold |
|---|---|---|---|
| A. Amplitude | pinball loss per quantile, CRPS | coverage of `[q05,q95]` | `skill > 0` vs CARR; coverage 90% ± 2 |
| A. σ_day | QLIKE on RV | Mincer-Zarnowitz `b=1` | `R²_OOS(log RV) > 0.35` |
| B. Asymmetry | log-loss | Brier skill vs 0.5 | `BSS > 0` |
| C. Timing | discrete log-score | KS of the PIT | KS < 0.05 (n ≈ 2,500) |
| D. Real time | **Brier skill vs diffusion** | ECE, calibration slope | `BSS > 0.05` |
| Economics | net P&L, Sharpe | max DD, turnover | see §8.4 |

### 8.4 Economics and overfitting

- **Costs:** 1.5 / 2.0 / 3.0 ticks round trip, as scenarios. A result that only exists at
  1 tick does not exist.
- **Block bootstrap** (block = 5 days) for the CI of the mean P&L: preserves
  autocorrelation.
- **SPA test** (Hansen 2005) or reality check (White 2000) when comparing several variants.
- **DSR** (Bailey & López de Prado 2014) with `n_trials` **actually recorded**. Without a
  trial ledger the DSR is incalculable and the result is indefensible.
- **PBO** (Bailey et al. 2017) by CSCV if you have fewer than 20 variants.
- **Decision rule:** a Sharpe of 1.0 over 500 trades has a 95% CI of about `[0.1; 1.9]`. It
  proves nothing. The number of *effective* trades matters, not the number of trades: with
  intraday correlated trades, divide by 3 to 5.

---

## 9. Complete architecture

```
                    ┌─────────────────────────────────────────┐
  J-1 16:05         │ POST-SESSION                            │
                    │ • RV, BV, range, P/GK/RS                 │
                    │ • label HOD/LOD, T_H, T_L, asymmetry     │
                    │ • update the F(t) profile (rolling)      │
                    │ • update HAR/CARR + calibration          │
                    └──────────────────┬──────────────────────┘
                                       ▼
                    ┌─────────────────────────────────────────┐
  J-1 17:00→09:15   │ PRE-OPEN                                │
                    │ • σ̂_day = HAR(log RV, HARQ) × exp(γ'X)  │
                    │ • σ̂_residual = σ̂_day·√(1−F(t))          │
                    │ • amplitude map q10..q90 (R⁺, R⁻)       │
                    │ • levels: PDH/PDL/PDC/ONH/ONL/VWAP      │
                    │ • P(trend up/down/range) ; is_event     │
                    └──────────────────┬──────────────────────┘
                                       ▼
  J 09:25           │ FROZEN PLAN                             │
                    │ expected H zone = O + [q50, q85]·σ̂      │
                    │ expected L zone = O − [q50, q85]·σ̂      │
                    │ no-trade band = |q25| on either side     │
                                       ▼
  J 09:30           │ OPEN                                    │
                    │ • gap, position vs ONH/ONL, vs PD VWAP  │
                    │ • if |gap| > 0.5σ̂ ⇒ re-weight the plan   │
                                       ▼
  J 10:00           │ INITIAL BALANCE                         │
                    │ • IBH/IBL available                     │
                    │ • Bayesian update of σ̂:                 │
                    │   σ̂² ← (1−w)·σ̂² + w·RV_30min/F(0.5h)    │
                    │   w ≈ 0.3 (weight of the 30-min signal) │
                                       ▼
  Every minute      │ REAL-TIME STATE                         │
                    │ • d_H = M_t − P_t ; d_L = P_t − m_t     │
                    │ • p_H = g(2Φ(d_H/S_t)−1, features)      │
                    │ • p_L = g(2Φ(d_L/S_t)−1, features)      │
                    │ • vol_ratio, dist_vwap, levels          │
                                       ▼
        ┌──────────────────────────────────────────────┐
        │ REVERSAL TRIGGER (short at H)                │
        │  p_H ≥ 0.80                                   │
        │  AND P_t ≥ M_t − 0.15·σ̂                       │
        │  AND confluence (real level OR VWAP) at 0.1σ̂  │
        │  AND rejection signature: new high made       │
        │     then close < M_t over 3 bars              │
        │  AND 10:00 ≤ t ≤ 15:30                        │
        │  AND vol_ratio ∈ [0.7 ; 2.0]  (no trend)      │
        └───────────────────┬──────────────────────────┘
                            ▼
        ┌──────────────────────────────────────────────┐
        │ RISK                                          │
        │ stop   = M_t + max(1.5·ATR5m, 0.25·σ̂)         │
        │ target = VWAP then q50 of the opposite branch │
        │ size   = risk_budget / stop_distance          │
        │ 1 attempt per level (no re-entry)             │
        │ max 2 trades/day, max loss 2R/day             │
        │ forced exit 15:50                             │
        └───────────────────┬──────────────────────────┘
                            ▼
                    ┌─────────────────────────────────────────┐
                    │ JOURNAL + MONITORING                    │
                    │ • Brier / calibration per bucket         │
                    │ • P&L per level, per hour, per regime    │
                    │ • kill switch if BSS < 0 over 20 sessions│
                    └─────────────────────────────────────────┘
```

**Three architectural decisions that matter more than the model:**

1. **One mathematical object** for remaining amplitude and survival probability (§A.8). Two
   separate models diverge.
2. **The Bayesian update of σ̂ at 10:00** is the most reliable gain of the whole system: the
   first 30 minutes' volatility carries more information about the rest of the session than
   the entire pre-open model.
3. **The kill switch is part of the system, not an option.** `BSS < 0` over 20 consecutive
   sessions ⇒ stop, not "parameter tuning".

---

## 10. Concrete test plan

| Step | Test | Failure criterion → action |
|---|---|---|
| **T0** | Data integrity: continuous stitching, `America/New_York` time zone, half sessions, missing minutes, duplicates | > 0.5% of sessions with a suspicious timestamp ⇒ fix before anything else |
| **T1** | Label audit: recompute H/L with a second aggregation; count the share of HOD in the last 5 minutes | gap > 1 tick ⇒ aggregation bug |
| **T2** | **Null replication:** arcsine law on `F(T_H)` (KS); half-normality of `R⁺` (multiplier) | KS < 0.05 expected; if KS > 0.15, the `F(t)` profile is badly estimated |
| **T3** | Amplitude: walk-forward, pinball vs CARR and vs half-normal | skill ≤ 0 ⇒ the model adds nothing, fall back to CARR |
| **T4** | Real time: Brier at 10:00/11:00/12:30/14:00/15:00, **vs the diffusion formula** | BSS ≤ 0 ⇒ no edge on the probability, stop here |
| **T5** | Levels: `delta` with matched placebo, Bonferroni correction over ~8 levels | `significant=False` ⇒ remove the confluence filter |
| **T6** | Economics: costs 1.5/2/3 ticks, block bootstrap, DSR with trial ledger | DSR < 0.95 ⇒ do not trade |
| **T7** | Stability: same P&L sign on ES and NQ, over 2 sub-periods, and monotone in the `p_H` threshold | non-monotone ⇒ threshold artefact, not an edge |
| **T8** | Paper trading for 3 months, minimum size | divergence of more than 50% from the backtest ⇒ the backtest is wrong |

**Test T2 is the most important and the cheapest.** If you cannot reproduce the arcsine law
and half-normality on your data, your pipeline has a bug, and no amount of ML will fix it.

---

## 11. Honest estimate of the edge

### What is solid

- **Forecasting volatility:** yes, plainly. `R²_OOS(log RV) ≈ 0.4–0.6`. Real, robust,
  replicated. But it is **size**, not **direction**.
- **The timing prior:** real, free, and… public and parameter-free. It brings no edge, it
  brings **better calibration**, which is already a lot if it stops you trading at the
  wrong moment.
- **The survival probability:** the object is correct and the formula exact under its
  assumptions. The calibration layer on top is the only place where an edge can appear.

### What is thin

- **Asymmetry:** the R² of the most solid published effect (Gao et al. 2018) is
  **1.4–2.0% out of sample**. Expect a shift of a few probability points, not a clean
  signal.
- **HOD reversal has negative skew:** you sell a new high in a day that may be a trend day.
  Losses are larger than gains, so the hit rate must be high to compensate.
- **The family base rate:** the 2026 pre-registered studies on ORB in ES/NQ find a median
  gross edge of **−0.01 tick**. That is the realistic starting point.

### Sizing

Assumptions: ES, `σ_day = 31 pts`, stop at `0.25σ ≈ 8 pts` (R = 8 pts = $400), target 1R,
cost 2 ticks = $25 = **6% of R**.

| Version | Trades/week | Gross expectancy/trade | Net | Expected annualised Sharpe |
|---|---|---|---|---|
| Naive (fade every touch of the σ band) | 8–12 | ≈ 0 | **negative** (−6% of R) | < 0, certain |
| Filtered (p_H ≥ 0.8 + confluence + rejection) | 2–4 | 0.05–0.15 R | 0 to +0.10 R | 0.2–0.6 |
| Filtered + regime (vol, event, trend) | 1–2 | 0.10–0.25 R | +0.05–0.20 R | 0.3–0.8 |

**What these numbers are worth:** with 300 trades, the 95% CI of the Sharpe is about
`[−0.1; 0.9]`. Statistically, 0.4 cannot be distinguished from 0. It takes **500 to 1,000
trades** (3 to 5 years at 2–4 per week) to decide.

**Probability that a careful implementation has a genuinely positive edge: 25–40%.** That is
not pessimism, it is the family base rate, costs included.

**Expected degradation from backtest to live: 40–60%** of the Sharpe. Causes: limit fills in
the wick (you are not alone selling the new high), slippage on trend days, and the fact that
the backtest assumes you are there for every signal.

### What would make me believe the result

1. `BSS > 0.05` against the diffusion formula, out of sample, over 500+ sessions.
2. The result survives **3 ticks** round trip.
3. **DSR > 0.95** with a recorded `n_trials`.
4. The same sign on **ES and NQ**, and over **two sub-periods**.
5. The effect is **monotone** in the `p_H` threshold (0.6 → 0.7 → 0.8 → 0.9), not a jump at
   a magic threshold.

If any of these five is missing, it is not an edge, it is a backtest.

---

## 12. Code

| File | Content |
|---|---|
| `hodlod/vol.py` | Parkinson, GK, RS, Yang-Zhang; seasonal profile; volatility time; HAR-RV (log, smearing); CARR(1,1) quasi-ML with BHHH standard errors; RV and BV |
| `hodlod/amplitude.py` | half-normal benchmark + multiplier; GPD/POT (fit, quantile, tail prob); quantile regression; `QuantileBooster`; crossing detection; asymmetry classifier |
| `hodlod/timing.py` | arcsine laws (cdf/pdf/ppf); volatility time; two-step PIT; KS; log-score; empirical hazard; competing risks |
| `hodlod/realtime.py` | `p_no_new_high` (driftless), `p_no_new_high_drift`, `p_range_within` (two barriers), stochastic-vol average, `remaining_sigma`, `session_features`, isotonic calibration |
| `hodlod/levels.py` | reference levels, initial balance, touch/break/reject, **matched-distance placebo**, `confluence_table` with bootstrap CI, approximate volume profile |
| `hodlod/evaluate.py` | `purged_walk_forward`, Brier/BSS/log-loss/ECE/AUC, pinball/CRPS/coverage, block bootstrap, PSR/DSR, `leakage_audit` |
| `hodlod/selftest.py` | 21 Monte-Carlo checks of every closed form |

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m hodlod.selftest      # 21/21
```

### What the tests caught (and why that is the most useful part)

| Bug | Consequence if undetected |
|---|---|
| Two-barrier series without the reflected images | `P(range)` overstated by 84% (0.68 instead of 0.37) |
| PIT tested against the uniform instead of the arcsine | systematic rejection of a correct model |
| CARR simulation without the `α` term | validating a memoryless model, and a false belief that CARR fails |
| Scalar boolean indexing (NumPy 2) | silent crash of the drift branch |
| Indentation broken by a patch | `ImportError` at import |

This is the argument for writing the tests before the backtest: of five bugs, **four were in
my own validation**, not in the models.

---

## 13. What I would do in the first 10 days

1. **Days 1–2:** T0 + T1. Clean the data. Nothing else.
2. **Day 3:** T2. Replicate the arcsine law and half-normality. If that fails, everything else
   is wrong.
3. **Days 4–5:** HAR + CARR, `σ̂_day`, QLIKE, `R²_OOS`. That is the foundation.
4. **Days 6–7:** amplitude map by quantile regression, pinball, coverage.
5. **Day 8:** raw diffusion `p_final` + isotonic calibration. **BSS against the formula.**
6. **Day 9:** T5, levels with placebo. This is the filter most likely to overstate its own
   power.
7. **Day 10:** decision. If the BSS from step 5 is ≤ 0, **stop** — there is no signal in the
   probability, and no level filter will create one.
