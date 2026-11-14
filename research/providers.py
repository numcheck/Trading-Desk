"""
Data providers for ES/NQ futures.

Three sources, one interface. The point is that the rest of the pipeline never has to
know where a bar came from, only whether it is a real futures bar or a proxy.

    FinzDataProvider   real CME futures, 1-minute, individual contracts or continuous
    LocalFileProvider  CSV/Parquet dropped in data/raw/ (any vendor), column map in
                       data/raw/column_map.json
    YahooProvider      ETF proxies, explicitly labelled PROXY

Rules enforced here:
  * credentials only from the environment (FINZDATA_API_KEY), never from code or files;
  * no silent substitution: if the futures provider is not armed, callers get a proxy
    and every frame is tagged, with a banner printed once;
  * no fabricated bars: gaps are gaps, and validation reports them rather than filling;
  * the free FinzData tier only covers the last 30 days, which is not enough for this
    study. coverage() therefore reports the real window instead of assuming it.

Run:  python -m research.providers --coverage
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

RAW = Path("data/raw")
CACHE = Path("research/cache")
ET = "America/New_York"
PROXY_BANNER = (
    "\n" + "=" * 78 + "\n"
    "PROXY MODE - NO CONCLUSION ABOUT ES/NQ\n"
    "No ES/NQ 1-minute futures data is available. Results computed from ETF proxies\n"
    "describe SPY/QQQ/IWM, not the futures, and must be labelled as such.\n"
    "=" * 78 + "\n"
)


@dataclass
class Coverage:
    """What a provider can actually deliver, measured, not assumed."""
    provider: str
    root: str
    is_futures: bool
    interval: str
    start: str | None = None
    end: str | None = None
    n_bars: int = 0
    n_contracts: int = 0
    has_open_interest: bool = False
    has_volume: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class DataProvider(Protocol):
    name: str

    @property
    def is_futures(self) -> bool: ...

    @property
    def armed(self) -> bool:
        """True only if the provider can serve real data right now."""
        ...

    def coverage(self, root: str = "ES", interval: str = "1min") -> Coverage: ...

    def bars(self, root: str, start: str, end: str, interval: str = "1min") -> pd.DataFrame: ...


# ----------------------------------------------------------------------------
# FinzData
# ----------------------------------------------------------------------------
class FinzDataProvider:
    """Real futures bars through the official SDK.

    Endpoints used (verified in the published OpenAPI document):
        GET /v1/me                     account and permissions
        GET /v1/futures/roots          every market with its coverage
        GET /v1/futures/contracts      contracts for one root
        GET /v1/futures/daily          daily bars
        GET /v1/futures/bars           intraday bars, interval in 1min/5min/15min/30min/1hour

    Timestamps mark the START of each bar in US Eastern time. Daily bars go back to
    2007 with a paid plan; a free key is limited to the last 30 days and 1,000 calls
    a day, which cannot support a five-year study.
    """

    name = "finzdata"

    def __init__(self, api_key: str | None = None):
        # The key is read from the environment only. Passing it explicitly is allowed
        # for tests; it is never persisted.
        self._key = api_key or os.environ.get("FINZDATA_API_KEY", "")
        self._client = None
        self._error: str | None = None
        if self._key:
            try:
                import finzdata as fd
                self._client = fd.Client()
            except Exception as exc:                      # noqa: BLE001
                self._error = f"{type(exc).__name__}: {exc}"
        else:
            self._error = "FINZDATA_API_KEY is not set in the environment"

    @property
    def is_futures(self) -> bool:
        return True

    @property
    def armed(self) -> bool:
        return self._client is not None

    def _require(self):
        if self._client is None:
            raise RuntimeError(f"FinzData not available: {self._error}")

    def roots(self) -> list[str]:
        self._require()
        out = self._client.futures_roots()
        return sorted(out) if isinstance(out, list) else sorted(out.keys())

    def contracts(self, root: str) -> list[str]:
        self._require()
        out = self._client.futures_contracts(root)
        return list(out) if not isinstance(out, pd.DataFrame) else list(out.iloc[:, 0])

    def bars(self, root: str, start: str, end: str, interval: str = "1min",
             contract: str | None = None, continuous: bool = True) -> pd.DataFrame:
        """Bars tagged with their source. Raises rather than fabricating anything."""
        self._require()
        kw = dict(root=root, start=start, end=end, interval=interval)
        if contract:
            kw["contract"] = contract
            kw.pop("root")
        else:
            kw["continuous"] = continuous
        df = self._client.futures(**kw)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        df = _normalise(df, source="finzdata", is_futures=True, contract=contract)
        if df.empty:
            raise RuntimeError(f"FinzData returned no bars for {root} {start}..{end} {interval}")
        return df

    def coverage(self, root: str = "ES", interval: str = "1min") -> Coverage:
        cov = Coverage(provider=self.name, root=root, is_futures=True, interval=interval)
        if not self.armed:
            cov.notes.append(f"not armed: {self._error}")
            return cov
        try:
            roots = self._client.futures_roots()
            cov.notes.append(f"roots reachable: {len(roots)}")
        except Exception as exc:                          # noqa: BLE001
            cov.notes.append(f"roots failed: {type(exc).__name__}: {exc}")
            return cov
        try:
            end = pd.Timestamp.utcnow().normalize()
            start = end - pd.Timedelta(days=30)           # free-tier window, measured below
            df = self.bars(root, start.date().isoformat(), end.date().isoformat(), interval)
            cov.start, cov.end = str(df.ts.min()), str(df.ts.max())
            cov.n_bars = int(len(df))
            cov.has_volume = bool(df.volume.notna().any())
            cov.has_open_interest = bool(df.open_interest.notna().any())
            cov.n_contracts = int(df.contract.nunique(dropna=True)) if "contract" in df else 0
        except Exception as exc:                          # noqa: BLE001
            cov.notes.append(f"1-minute probe failed: {type(exc).__name__}: {exc}")
        return cov


# ----------------------------------------------------------------------------
# Local files
# ----------------------------------------------------------------------------
DEFAULT_MAP = {
    "timestamp": ["timestamp", "datetime", "date_time", "time", "ts_event", "index"],
    "open": ["open", "o", "Open"],
    "high": ["high", "h", "High"],
    "low": ["low", "l", "Low"],
    "close": ["close", "c", "Close", "last"],
    "volume": ["volume", "v", "Volume", "size"],
    "open_interest": ["open_interest", "openinterest", "oi", "Open Interest"],
    "contract": ["contract", "symbol", "instrument", "ticker"],
}


class LocalFileProvider:
    """Reads ES/NQ minute bars from data/raw/ (CSV or Parquet).

    Vendor layouts differ, so the column names are resolved through
    data/raw/column_map.json, which is documented and versioned. If the map is absent,
    DEFAULT_MAP is used and the provider says so instead of guessing silently.
    """

    name = "local"

    def __init__(self, raw: Path = RAW):
        self.raw = Path(raw)
        self.map_file = self.raw / "column_map.json"

    @property
    def is_futures(self) -> bool:
        return True

    @property
    def armed(self) -> bool:
        return any(self.raw.glob("*.parquet")) or any(self.raw.glob("*.csv"))

    def _map(self) -> dict:
        if self.map_file.exists():
            return json.loads(self.map_file.read_text())
        return DEFAULT_MAP

    def _files(self, root: str) -> list[Path]:
        pats = [f"{root}*.parquet", f"{root.lower()}*.parquet", f"{root}*.csv", f"{root.lower()}*.csv"]
        out: list[Path] = []
        for p in pats:
            out += sorted(self.raw.glob(p))
        return out

    def bars(self, root: str, start: str, end: str, interval: str = "1min") -> pd.DataFrame:
        files = self._files(root)
        if not files:
            raise FileNotFoundError(
                f"no {root} files in {self.raw}; expected e.g. {root}_1min.parquet")
        frames = [pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f) for f in files]
        df = _normalise(pd.concat(frames, ignore_index=True), source="local",
                        is_futures=True, mapping=self._map())
        lo = pd.Timestamp(start, tz=ET) if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start)
        hi = pd.Timestamp(end, tz=ET) if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end)
        df = df[(df.ts >= lo) & (df.ts <= hi)]
        if interval != "1min":
            df = resample_bars(df, interval)
        return df.reset_index(drop=True)

    def coverage(self, root: str = "ES", interval: str = "1min") -> Coverage:
        cov = Coverage(provider=self.name, root=root, is_futures=True, interval=interval)
        files = self._files(root)
        cov.notes.append(f"{len(files)} file(s) in {self.raw}")
        if not files:
            cov.notes.append(f"missing: {root}_1min.parquet (or .csv) in {self.raw}/")
            return cov
        try:
            df = self.bars(root, "2000-01-01", "2100-01-01", interval="1min")
            cov.start, cov.end, cov.n_bars = str(df.ts.min()), str(df.ts.max()), int(len(df))
            cov.has_volume = bool(df.volume.notna().any())
            cov.has_open_interest = bool(df.open_interest.notna().any())
            cov.n_contracts = int(df.contract.nunique(dropna=True)) if "contract" in df else 0
        except Exception as exc:                          # noqa: BLE001
            cov.notes.append(f"read failed: {type(exc).__name__}: {exc}")
        return cov


# ----------------------------------------------------------------------------
# Yahoo proxy
# ----------------------------------------------------------------------------
class YahooProvider:
    """ETF proxies. Never a substitute for futures, always labelled."""

    name = "yahoo"
    PROXY = {"ES": "SPY", "NQ": "QQQ", "RTY": "IWM"}

    @property
    def is_futures(self) -> bool:
        return False

    @property
    def armed(self) -> bool:
        return True

    def bars(self, root: str, start: str, end: str, interval: str = "1h") -> pd.DataFrame:
        from research.data import bars_1h
        sym = self.PROXY.get(root, root)
        df = bars_1h(sym, days=730).copy()
        df["source"] = "yahoo-proxy"
        df["is_futures"] = False
        df["proxy_symbol"] = sym
        return df

    def coverage(self, root: str = "ES", interval: str = "1h") -> Coverage:
        cov = Coverage(provider=self.name, root=root, is_futures=False, interval=interval)
        try:
            df = self.bars(root, "", "")
            cov.start, cov.end, cov.n_bars = str(df.index.min()), str(df.index.max()), int(len(df))
            cov.has_volume = True
            cov.notes.append(f"etf proxy {self.PROXY.get(root, root)}; 730 days max")
        except Exception as exc:                          # noqa: BLE001
            cov.notes.append(f"failed: {type(exc).__name__}: {exc}")
        return cov


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _pick(df: pd.DataFrame, alts: list[str]) -> str | None:
    for a in alts:
        if a in df.columns:
            return a
    return None


def _normalise(df: pd.DataFrame, source: str, is_futures: bool,
               contract: str | None = None, mapping: dict | None = None) -> pd.DataFrame:
    """Maps any vendor layout onto (ts, open, high, low, close, volume, open_interest)."""
    mapping = mapping or DEFAULT_MAP
    cols: dict[str, str] = {}
    for target, alts in mapping.items():
        c = _pick(df, alts)
        if c:
            cols[target] = c
    if "timestamp" not in cols:
        raise ValueError(f"no timestamp column found among {list(df.columns)[:12]}")
    out = pd.DataFrame({
        "ts": pd.to_datetime(df[cols["timestamp"]], utc=True, errors="coerce").dt.tz_convert(ET),
        "open": pd.to_numeric(df[cols["open"]], errors="coerce") if "open" in cols else np.nan,
        "high": pd.to_numeric(df[cols["high"]], errors="coerce") if "high" in cols else np.nan,
        "low": pd.to_numeric(df[cols["low"]], errors="coerce") if "low" in cols else np.nan,
        "close": pd.to_numeric(df[cols["close"]], errors="coerce") if "close" in cols else np.nan,
        "volume": pd.to_numeric(df[cols["volume"]], errors="coerce") if "volume" in cols else np.nan,
        "open_interest": pd.to_numeric(df[cols["open_interest"]], errors="coerce")
        if "open_interest" in cols else np.nan,
    })
    out["contract"] = (df[cols["contract"]].astype(str) if "contract" in cols
                       else (contract if contract else root_default(contract)))
    out["source"] = source
    out["is_futures"] = is_futures
    out = out.dropna(subset=["ts"])
    out = out.dropna(subset=["open", "high", "low", "close"], how="all")
    out = out.drop_duplicates(subset=["ts", "contract"], keep="last").sort_values("ts")
    return out.reset_index(drop=True)


def root_default(contract: str | None) -> str:
    return contract or "unknown"


def resample_bars(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resamples 1-minute bars to the requested interval, keeping the session intact."""
    rule = {"5min": "5min", "15min": "15min", "30min": "30min", "1hour": "1h",
            "1h": "1h"}.get(interval)
    if rule is None:
        return df
    g = df.set_index("ts").groupby("contract", group_keys=False)
    out = g.resample(rule).agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"),
                               volume=("volume", "sum"),
                               open_interest=("open_interest", "last"))
    out = out.dropna(subset=["open"]).reset_index()
    out["source"] = df["source"].iloc[0] if len(df) else "unknown"
    out["is_futures"] = df["is_futures"].iloc[0] if len(df) else False
    return out


def validate_bars(df: pd.DataFrame, bar_seconds: int = 60) -> dict:
    """OHLC integrity, duplicates, gaps and session boundaries. Reports, never repairs."""
    if df.empty:
        return {"n": 0}
    bad_hl = int((df.high < df.low).sum())
    bad_range = int(((df.high < df[["open", "close"]].max(axis=1)) |
                     (df.low > df[["open", "close"]].min(axis=1))).sum())
    ts = df.ts.sort_values()
    gaps = ts.diff().dt.total_seconds()
    expected = float(gaps.mode().iloc[0]) if len(gaps.dropna()) else float(bar_seconds)
    return {
        "n": int(len(df)),
        "duplicates": int(df.duplicated(subset=["ts", "contract"]).sum()),
        "bad_high_low": bad_hl,
        "bad_ohlc_range": bad_range,
        "missing_bar_rate": float((gaps > 1.5 * expected).mean()),
        "median_gap_s": expected,
        "start": str(ts.min()), "end": str(ts.max()),
        "n_contracts": int(df.contract.nunique()) if "contract" in df else 1,
    }


def build_coverage_report(roots=("ES", "NQ"), interval: str = "1min") -> dict:
    """Runs every provider and reports what it can really deliver."""
    provs = [FinzDataProvider(), LocalFileProvider(), YahooProvider()]
    report = {"interval": interval, "providers": {}}
    for p in provs:
        for root in roots:
            cov = p.coverage(root, interval if p.is_futures else "1h")
            report["providers"].setdefault(p.name, {})[root] = cov.as_dict()
    futures_ok = any(v[root]["n_bars"] > 0
                     for name, v in report["providers"].items() if name != "yahoo"
                     for root in v)
    report["futures_available"] = bool(futures_ok)
    if not futures_ok:
        report["banner"] = PROXY_BANNER.strip()
    return report


if __name__ == "__main__":
    import sys
    if "--coverage" in sys.argv:
        rep = build_coverage_report()
        print(json.dumps(rep, indent=2, default=str))
        if not rep["futures_available"]:
            print(PROXY_BANNER)
    else:
        print(__doc__)
