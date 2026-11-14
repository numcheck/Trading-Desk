"""
Append-only trial registry.

Every evaluated configuration must be logged here, *including the ones that failed*.
Without this file the deflated Sharpe ratio cannot be computed honestly, and any
reported performance is indefensible. Deleting a bad trial is the single most effective
way to fake a backtest, so this module only ever appends.

The ledger lives at research/out/registry.jsonl, one JSON object per line:

    {"ts": "...", "phase": "1.4", "strategy": "fade", "instrument": "ES",
     "params": {...}, "n_trades": 558, "net_mean_R": -0.0035, "sharpe": 0.0238,
     "source": "research.engine", "note": "..."}

Usage:
    from research.registry import log_trial, read_ledger, n_trials
    log_trial(strategy="fade", instrument="ES", params={"theta": 0.7, "k": 1.0},
              n_trades=120, net_mean_R=0.01, sharpe=0.2)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path(os.environ.get("HODLOD_REGISTRY", "research/out/registry.jsonl"))


def log_trial(strategy: str, instrument: str, params: dict, n_trades: int,
              net_mean_R: float, sharpe: float | None = None, phase: str = "",
              source: str = "", note: str = "") -> dict:
    """Appends one trial. Never overwrites, never rewrites existing lines."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "phase": phase, "strategy": strategy, "instrument": instrument,
        "params": params, "n_trades": int(n_trades),
        "net_mean_R": None if net_mean_R is None else float(net_mean_R),
        "sharpe": None if sharpe is None else float(sharpe),
        "source": source, "note": note,
    }
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return row


def read_ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def n_trials(strategy: str | None = None) -> int:
    rows = read_ledger()
    if strategy is None:
        return len(rows)
    return sum(1 for r in rows if r.get("strategy") == strategy)


def sharpe_dispersion(strategy: str | None = None) -> float:
    """Standard deviation of the per-trial Sharpe, the sr_std that DSR needs."""
    import numpy as np
    rows = [r for r in read_ledger() if r.get("sharpe") is not None]
    if strategy is not None:
        rows = [r for r in rows if r.get("strategy") == strategy]
    s = np.array([r["sharpe"] for r in rows], float)
    return float(np.std(s, ddof=1)) if len(s) > 1 else 0.0


def seed_from_results(results_path: str = "research/out/results.json") -> int:
    """Imports the sweep already stored in results.json, once, without duplicating.

    The sweep is the 30-configuration grid per instrument that the engine already ran.
    Logging it is what makes the DSR honest: those are real trials, even though they
    were run before this registry existed.
    """
    p = Path(results_path)
    if not p.exists():
        return 0
    res = json.loads(p.read_text())
    seen = {(r.get("strategy"), r.get("instrument"),
             json.dumps(r.get("params"), sort_keys=True)) for r in read_ledger()}
    added = 0
    for row in res.get("sweep", []):
        params = {"theta": row["theta"], "k_tp": row["k_tp"]}
        key = ("fade", row["symbol"], json.dumps(params, sort_keys=True))
        if key in seen:
            continue
        log_trial(strategy="fade", instrument=row["symbol"], params=params,
                  n_trades=row.get("n", 0), net_mean_R=row.get("mean_R"),
                  sharpe=row.get("sharpe_trade"), phase="1.4",
                  source="research.engine.sweep",
                  note="in-sample sweep, proxy instrument")
        seen.add(key)
        added += 1
    return added


if __name__ == "__main__":
    added = seed_from_results()
    print(f"seeded {added} trials; ledger now holds {n_trials()} rows -> {LEDGER}")
    print(f"sharpe dispersion: {sharpe_dispersion():.4f}")
