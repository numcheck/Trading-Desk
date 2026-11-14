# State of work — alpha-research

Living handoff note. Updated whenever the work stops mid-flight. **Read this first.**

Branch: `alpha-research` (pushed to `origin`, PR link:
https://github.com/numcheck/Trading-Desk/pull/new/alpha-research). `main` untouched.

## Commits

| SHA | Phase | What it does |
|---|---|---|
| `f54373d` | 0 | Full translation to English; `report/HOD_LOD_rapport.md` → `HOD_LOD_report.md` |
| `2f416ef` | 1.1 | Sigma-to-points uses the current open, not the same day's close |
| `254c25b` | 1.2 | One entry per instrument per day (`MAX_PER_DAY`, `pick(..., max_per_day)`) |
| `aa70451` | 1.3 | Chronological metrics; `mc_stats` → `hodlod.evaluate.block_bootstrap_ci` on daily P&L |
| `20d016c` | 1.4 | `INSTRUMENTS` + `cost_scenarios`: ES/NQ 1.5/2/3 ticks, +1 tick on stop exits |
| `2f3a276` | 1.7 | matplotlib, reportlab, pillow, pytest in `requirements.txt` |
| `563ec24` | 1.8 | `tests/test_validation.py`, 10 tests |
| `6b1ff21` | — | README + theory report refreshed with post-fix numbers |

## Current measured result (proxy mode, SPY/QQQ/IWM hourly)

After commit 1.5 (HAR now includes the smearing correction, sigma ~8% higher):

```
n = 558   hit = 48.7%   gross (see results.json)   net = -0.0035 R
total = -1.9 R   max DD = 25.7 R   PF = 0.99
DSR = 0.008   permutation p = 0.539   block-bootstrap CI [-51.3, +41.6] R
Cost scenarios, ES: +0.0004 / -0.0074 / -0.0229 R at 1.5 / 2 / 3 ticks
Cost scenarios, NQ: +0.0245 / +0.0237 / +0.0221 R
```

Before commit 1.5 (smearing omitted, sigma understated by sqrt(smear) ~ 8%): n=521,
gross +0.0496, net +0.0194. **The smearing fix alone flips the sign.** That is the single
most consequential numerical finding of Phase 1 and it must be carried into the README.

Verdict: net expectancy is **negative and indistinguishable from zero**; Phase 5 criterion
(2) fails at 2 ticks already on ES.

## Phase status

| Phase | Status |
|---|---|
| 0 Translation | **Done**, verified 21/21 and identical numbers |
| 1.1–1.4, 1.7, 1.8 | **Done** |
| 1.5 Use hodlod | **PATCHED BUT UNCOMMITTED** — see below |
| 1.6 BSS vs diffusion at fixed times | Not started |
| 2 Data layer | Not started |
| 3 Pre-registration + registry + features | Not started |
| 4 Modelling | Not started |
| 5 Vault + kill criteria | Not started |

## Immediate next step (uncommitted work)

`research/engine.py` currently has the item-1.5 patch applied **in the working tree**:

- imports `brier`, `brier_skill`, `p_final_now`, `HAR`
- `har_walkforward` delegates to `hodlod.vol.HAR`
- `realtime_rows` uses `p_final_now`
- `brier_table` uses `brier_skill`

The engine run was killed at the 5-minute cap (exit 137). **Cause to verify:** the new
`har_walkforward` calls `model.predict(var[:t])` for every `t`, i.e. an O(n²) pandas
rebuild. The intended fix is to predict only once per refit window:

```python
model = HAR().fit(var[:t])
refit_at = t + step
nxt = min(refit_at, n)
preds = model.predict(var[:nxt])     # rows t..nxt-1 use features <= their own index-1
out[t:nxt] = np.sqrt(np.clip(preds[t:nxt], 0, None))
```

This is still causal: coefficients are frozen at `t`, and the design row for index `j`
only uses lags of `j` (<= j-1).

Then: run `python -m research.engine`, compare headline numbers against the table above
(the refactor is meant to be numerically near-identical), and commit item 1.5 separately.

## Gotchas discovered

- `pytest` was missing from the venv: `.venv/bin/pip install pytest`.
- The venv is `--system-site-packages`; installing into system site-packages fails with
  Errno 13, always install into `.venv`.
- `research/engine.py` imports `research.data`, which reads the pickle cache in
  `research/cache/` if present. The cache exists, so re-runs take seconds **when the
  forecast loop is not O(n²)**.
- `results.json` field names consumed by `research/report.py`: `oos`, `mc.boot_ci`,
  `mc.boot_mean`, `mc.perm_p`, `mc.boot_dist`, `mc.perm_dist`, `dsr`, `cost_scenarios`,
  `freq_signal`, `sweep`, `amplitude`, `timing`, `realtime`, `freq_amp`, `meta`.
  Changing them breaks the PDF build.
- The French accents were fine, but `σ̂` (sigma + combining circumflex) renders as "ó" in
  both matplotlib and reportlab. Use plain `σ` everywhere.

## Credentials

`FINZDATA_API_KEY` was **not** present in the environment. A key was pasted into the chat
at some point; it must not be written to any file, and it should be treated as leaked
(rotate it). No credential is referenced in the code.

## Remaining work, in order

1. Fix and commit 1.5 (above). Then 1.6.
2. Phase 2: `DataProvider` protocol, `LocalParquetProvider` (data/raw, column mapping
   file), `YahooProvider` (proxy mode with a loud banner), contract roll by volume,
   RTH + overnight sessions, `bar_minutes` and `slots` as parameters, Parquet cache,
   coverage report, `python -m research.data --coverage`.
3. Phase 3: `research/PREREGISTRATION.md` (committed **before** any feature test),
   `research/out/registry.jsonl` append-only writer replacing `trial_ledger`,
   feature groups A–F behind availability flags.
4. Phase 5: vault of the last 12 months, killed by a frozen commit hash, and the six
   acceptance criteria written down before Phase 4 touches real ES/NQ data.
5. Phase 4: elastic net then shallow HistGradientBoosting, purged monthly walk-forward,
   threshold-response curve, baselines (raw fade, time/side-matched random, shuffled
   features, buy-and-hold).
6. Regenerate the PDF and push `alpha-research` again.

## Reproduce

```bash
.venv/bin/python -m hodlod.selftest     # 21/21
.venv/bin/python -m pytest tests -q     # 10 passed
.venv/bin/python -m research.engine     # study + backtest + Monte-Carlo
.venv/bin/python -m research.report     # PDF + figures
```

## Phase 2 — started (uncommitted caveat resolved into this note)

`research/providers.py` now exists: `DataProvider` protocol, `FinzDataProvider`
(official SDK, `/v1/futures/{roots,contracts,daily,bars}`, key from `FINZDATA_API_KEY`
only), `LocalFileProvider` (`data/raw/`, column map in `data/raw/column_map.json`),
`YahooProvider` (proxy, loud banner), `resample_bars`, `validate_bars`,
`build_coverage_report`, and `python -m research.providers --coverage`.

**Known defect, not fixed:** the `--coverage` output repeats the PROXY banner several
times instead of once. The JSON payload is correct; the printing path is wrong. Fix by
printing the report first and the banner once, outside any formatting helper.

**Verified against the live API, without credentials:**
`GET https://api.finzdata.com/v1/me` returns 401 with
`invalid or missing API key`. The published OpenAPI document lists
`/v1/futures/roots`, `/v1/futures/contracts?root=`, `/v1/futures/daily?root=&continuous=`,
`/v1/futures/bars?root=&interval=`. Intraday is available from February 2007 for 139
markets, timestamps are bar START in US Eastern.
**A free key covers only the last 30 days and 1,000 calls/day**, which cannot support
the five-year (2021→present) requirement. Five years needs a paid plan.

**Not done in Phase 2:** contract roll by volume with recorded roll dates, RTH +
overnight session construction from 1-minute bars, DST/holiday/half-day calendar,
`bar_minutes`/`slots` parameterisation of the engine (still 7 hourly slots), Parquet
cache for futures bars.
