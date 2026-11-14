# Pre-registration

Frozen before any feature is evaluated. **Append-only**: later changes are added at the
bottom with a date and a reason, never by editing what is above.

Status of the data when this was written: **PROXY MODE — NO CONCLUSION ABOUT ES/NQ.**
Available: ES/NQ daily 2021-05→2026-11 (1,383 sessions), SPY/QQQ/IWM hourly 2024-11→2026-11
(491 sessions). One-minute ES/NQ futures are not available: the FinzData free tier covers
30 days, which cannot support the required five-year window. Every result produced while
this banner applies describes the ETFs, not the futures.

## 0. What is being tested

The base model fades a session extreme when the survival probability
`p = 2*Phi(d/S) - 1` exceeds `theta`. Under a martingale this has **zero gross
expectancy by construction**, which is why the raw fade is the baseline and not the
candidate. The hypothesis under test is narrower:

> **H1.** Conditional on context available at the decision time, the fade has a positive
> net expectancy in R after costs.

H1 is falsified by any of the criteria in section 3. A secondary hypothesis:

> **H2.** A range-bound day and a trend day have different optimal actions: fade on the
> former, continue on the latter.

## 1. Features, with their expected sign

Sign is the predicted effect on **net R of the fade** (positive = the fade does better).
Signs are written now so that a fitted coefficient contradicting them is a finding, not a
tuning opportunity. Every feature uses only information available at the decision bar.

### A. Day type

| Feature | Definition | Rationale | Expected sign |
|---|---|---|---|
| `or30_sigma` | high−low of the first 30 min, over `sigma_day` | an opening drive that consumes variance leaves less for a reversal | **−** |
| `or60_sigma` | same over 60 min | as above, smoother | **−** |
| `gap_sigma` | (open − previous RTH close) / `sigma_day` | a large gap is often a repricing, not noise | **−** (fade worse) |
| `gap_fill_frac` | fraction of the gap already retraced | a filled gap removes the impulse to continue | **+** |
| `pos_in_on_range` | (price − ON low)/(ON high − ON low) | position inside the overnight range | 0 (control) |
| `rv_over_expected` | realized RV so far / (`sigma_day²·F(t)`) | >1 means the day is running hot | **−** |
| `efficiency_ratio` | \|close − open\| / sum\|Δclose\| | trending vs choppy | **−** |
| `vwap_crossings` | number of VWAP crossings since the open | many crossings = chop = fade works | **+** |
| `dist_vwap_sigma` | (price − VWAP)/`sigma_day` | stretch from the session's fair value | **+** (stretched = revert) |
| `F_now` | expected variance share elapsed | a late decision has less room left | **+** |
| `bars_since_extreme` | bars since the last new session extreme | a stale extreme is more likely final | **+** |

### B. Events

| Feature | Definition | Rationale | Expected sign |
|---|---|---|---|
| `is_fomc` | FOMC decision day (14:00 ET) | scheduled shock breaks the fade | **−** |
| `is_cpi`, `is_nfp` | 08:30 ET releases | ditto, pre-open | **−** |
| `is_opex` | monthly/quarterly expiry | pinning, then release | 0 / ambiguous |
| `is_month_end`, `is_quarter_end` | last session of the period | rebalancing flows | **−** |
| `is_half_day` | 13:00 close | different `F(t)`, thin afternoon | **−** |
| `is_pre_holiday`, `is_post_holiday` | adjacent to a market holiday | thin, gap-prone | **−** |

Dates come from `data/calendars/*.csv` with a source URL per row. **Unverified dates are
omitted, never guessed.** Missing files are reported by the loader, not filled.

### C. Cross-market

| Feature | Definition | Rationale | Expected sign |
|---|---|---|---|
| `nonconfirmation` | ES made a new extreme, NQ did not (or vice versa) | a single-index extreme is more likely to fail | **+** |
| `rel_strength_z` | z-score of ES−NQ return since the open | leadership vs lag | ambiguous |
| `vix_chg_open` | change in VIX since the open | risk appetite shift | **−** when rising |

Requires a second index. Available through the Yahoo path (SPY vs QQQ); RTY/YM only if the
series exist.

### D. Options (optional, needs option data)

`gamma_flip_dist`, `net_gamma_sign` proxied from open interest. Model-based, tested as a
hypothesis like any other. **Not implemented**: no option chain data available. Flagged off.

### E. Volume microstructure (optional)

`vol_at_extreme_z` (volume at the extreme vs its time-of-day expectation),
`absorption` (high volume, low range progress), `rejection_wick` (fixed rule: upper wick
≥ 60% of bar range and close in the bottom third), `time_beyond_prev_extreme`.
Needs volume; hourly ETF volume is a poor proxy, so this is **flagged off** until futures
data exists.

### F. Level confluence

`PDH`, `PDL`, `ONH`, `ONL`, `IBH`, `IBL`, `VWAP`, `POC`, round numbers, via
`hodlod.levels.reference_levels` / `initial_balance` / `volume_profile_poc`.
Each is evaluated **against a matched-distance placebo** (`placebo_levels`,
`confluence_table`) with a Bonferroni correction over the number of levels tested.
Expected sign: **+** if real levels reject more often than the placebo; otherwise the
level is decoration.

## 2. Model specification, fixed in advance

- Primary: fade candidates, `p` calibrated walk-forward with `IsotonicCalibrator`,
  triple-barrier outcomes (stop, `k·R`, session close), costs applied,
  **at most one trade per instrument per day**.
- Threshold grid: `theta ∈ {0.4, 0.5, 0.6, 0.7, 0.8}` × `k ∈ {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}`.
  Nothing outside this grid is tested. **30 combinations per instrument.**
- Secondary: elastic net on **at most 10 features**, expanding window, monthly refit,
  purged with `purged_walk_forward(embargo=5)`. Then, only if the elastic net shows
  signal, a `HistGradientBoostingRegressor(max_depth=3)` with monotonic constraints
  matching the signs in section 1. **No feature selection on test data.**
- Trade only when predicted net R exceeds a pre-registered threshold; the full
  threshold-response curve is reported, not just the best point.
- Baselines that must all be beaten: raw fade; random entries matched on time of day,
  side and holding time; the model with shuffled features; buy-and-hold over the same
  sessions.

## 3. Kill criteria — frozen

An edge may be claimed **only if all six hold**. Any failure stops the work: no new
variant is added, and a short negative report is written instead.

1. **DSR > 0.95** using the real trial count from `research/out/registry.jsonl`.
2. **Net expectancy > 0 in the 3-tick scenario** on ES.
3. **Same sign on ES and NQ**, and on **two disjoint sub-periods**.
4. **Monotone response** in the filter threshold (no jump at a magic value).
5. **BSS > 0.05** of the calibrated probability against the **raw diffusion formula**,
   out of sample, over **at least 500 sessions**.
6. The **vault** result (last 12 months, evaluated once, commit hash recorded) is
   consistent with the walk-forward bootstrap interval.

Criteria may never be loosened after seeing results.

## 4. Current status against these criteria

Evaluated on proxies, so **none of it counts as evidence about ES/NQ**, but the arithmetic
is already decisive:

| Criterion | Current value | Pass? |
|---|---|---|
| 1. DSR > 0.95 | 0.008 | ❌ |
| 2. net > 0 at 3 ticks | −0.0229 R | ❌ |
| 3. same sign ES/NQ | not measurable (proxies) | — |
| 4. monotone in theta | not measured | — |
| 5. BSS > 0.05 vs diffusion | −0.08 at 10:00, +0.17 at 12:30, ~200 sessions | ❌ |
| 6. vault | not opened | — |

**Verdict: H1 is not supported.** The fade has negative net expectancy, and the sign of
the result proved unstable under two corrections that had nothing to do with the strategy
(a look-ahead in the sigma conversion, and a missing smearing factor).

---

## Append log

- **2026-11-14** — created, before any feature was evaluated. Data: proxy mode.
