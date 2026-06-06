"""Overnight ETF basket backtester.

This is the first, deliberately simple engine for Txtadel-style close-to-open
research. It does not try to infer the hidden selection rule yet. It answers a
cleaner question first: if we hold a specified liquid ETF basket from today's
close to the next session's open, what does the P&L distribution look like?
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd


DEFAULT_UNIVERSE = ["XLU", "GLD", "EMXC", "XLE", "SMH", "CGDV", "XLI", "AVUV"]
TRADING_DAYS = 252


def _local_config_value(name, default=None):
    """Read optional local config without requiring config_local.py."""
    try:
        import config_local as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def normalize_weights(weights, tickers):
    """Return a long-only weight Series aligned to tickers and summing to 1."""
    if weights is None:
        return pd.Series(1.0 / len(tickers), index=list(tickers), dtype=float)
    w = pd.Series(weights, dtype=float).reindex(list(tickers)).fillna(0.0)
    w = w[w > 0]
    if w.empty:
        raise ValueError("weights must contain at least one positive ticker weight")
    return w / w.sum()


def _clean_ohlc_frame(df):
    """Normalize one ticker OHLC frame to date index and Open/Close columns."""
    out = df.loc[:, ["Open", "Close"]].copy()
    out = out.dropna(how="any")
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out.sort_index()


def load_daily_ohlc_yfinance(tickers, start, end):
    """Load adjusted daily Open/Close via yfinance.

    yfinance's adjusted mode scales Open and Close consistently, which keeps
    close-to-open returns split-adjusted.
    """
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    bars = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if ("Open", t) in raw.columns and ("Close", t) in raw.columns:
                bars[t] = _clean_ohlc_frame(pd.DataFrame({
                    "Open": raw[("Open", t)],
                    "Close": raw[("Close", t)],
                }))
    else:
        if len(tickers) != 1:
            raise ValueError("unexpected yfinance shape for multiple tickers")
        bars[list(tickers)[0]] = _clean_ohlc_frame(raw)
    return {k: v for k, v in bars.items() if not v.empty}


def _polygon_daily_ohlc(ticker, start, end, api_key):
    params = urllib.parse.urlencode(dict(adjusted="true", sort="asc", limit=50000, apiKey=api_key))
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["Open", "Close"])
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return pd.DataFrame({"Open": df["o"].astype(float).values,
                         "Close": df["c"].astype(float).values}, index=idx)


def load_daily_ohlc_polygon(tickers, start, end, api_key=None):
    """Load adjusted daily Open/Close via Polygon/Massive."""
    key = api_key or os.getenv("POLYGON_API_KEY") or _local_config_value("POLYGON_API_KEY")
    if not key:
        raise ValueError("POLYGON_API_KEY is not configured")
    bars = {t: _clean_ohlc_frame(_polygon_daily_ohlc(t, start, end, key)) for t in tickers}
    return {k: v for k, v in bars.items() if not v.empty}


def load_daily_ohlc(tickers=DEFAULT_UNIVERSE, start="2019-01-01", end=None, provider="auto", api_key=None):
    """Load daily Open/Close bars. auto prefers Polygon, then yfinance."""
    provider = provider.lower()
    if provider not in {"auto", "polygon", "yfinance"}:
        raise ValueError("provider must be auto, polygon, or yfinance")
    if provider in {"auto", "polygon"}:
        try:
            bars = load_daily_ohlc_polygon(tickers, start, end, api_key=api_key)
            if bars:
                return bars, "polygon"
        except Exception:
            if provider == "polygon":
                raise
    bars = load_daily_ohlc_yfinance(tickers, start, end)
    return bars, "yfinance"


def overnight_returns(bars):
    """Return close-to-next-open returns indexed by the close date.

    A row dated 2026-06-04 means buy that ticker at the 2026-06-04 close and
    sell at the next available session's open.
    """
    series = {}
    for ticker, df in bars.items():
        b = _clean_ohlc_frame(df)
        r = b["Open"].shift(-1) / b["Close"] - 1.0
        series[ticker] = r.dropna()
    return pd.DataFrame(series).sort_index()


def backtest_static_basket(overnight, weights=None, cost_bps=0.0):
    """Backtest a fixed overnight basket.

    net subtracts one round-trip cost per overnight basket trade. If a ticker is
    missing on a date, the available weights are renormalized for that date.
    """
    if overnight.empty:
        return pd.DataFrame(columns=["date", "gross", "net", "n_assets"])
    tickers = list(overnight.columns)
    base_w = normalize_weights(weights, tickers)
    rows = []
    for date, row in overnight.iterrows():
        available = row.dropna()
        w = base_w.reindex(available.index).dropna()
        if available.empty or w.empty:
            continue
        w = w / w.sum()
        gross = float((available.loc[w.index] * w).sum())
        rows.append(dict(date=date, gross=gross, net=gross - cost_bps / 1e4,
                         n_assets=int(len(w))))
    return pd.DataFrame(rows)


def performance_metrics(res, benchmark=None):
    """Compute compact daily-return performance metrics."""
    if res.empty:
        return dict(n_days=0, total_return_pct=np.nan, cagr_pct=np.nan,
                    ann_vol_pct=np.nan, sharpe=np.nan, max_drawdown_pct=np.nan,
                    win_rate_pct=np.nan, mean_bps=np.nan, benchmark_excess_pct=np.nan)
    r = res["net"].astype(float)
    equity = (1.0 + r).cumprod()
    total = float(equity.iloc[-1] - 1.0)
    years = len(r) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    sharpe = float(r.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else np.nan
    peak = equity.cummax()
    dd = float(((peak - equity) / peak).max())
    out = dict(
        n_days=int(len(r)),
        total_return_pct=total * 100,
        cagr_pct=cagr * 100,
        ann_vol_pct=vol * 100,
        sharpe=sharpe,
        max_drawdown_pct=dd * 100,
        win_rate_pct=float((r > 0).mean() * 100),
        mean_bps=float(r.mean() * 1e4),
        benchmark_excess_pct=np.nan,
    )
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(res["date"]).dropna()
        aligned = res.set_index("date").loc[b.index, "net"]
        if not aligned.empty:
            out["benchmark_excess_pct"] = float(((1 + aligned).prod() - (1 + b).prod()) * 100)
    return out


def benchmark_close_to_close(bars, ticker="SPY"):
    """Compute close-to-close benchmark returns from a bars dict."""
    if ticker not in bars:
        return pd.Series(dtype=float, name=ticker)
    b = _clean_ohlc_frame(bars[ticker])
    return b["Close"].pct_change().dropna().rename(ticker)


def print_report(res, metrics, title="overnight basket"):
    """Print a clean text report."""
    print(f"== {title} ==")
    if res.empty:
        print("No trades produced.")
        return
    print(f"days: {metrics['n_days']}")
    print(f"mean/trade: {metrics['mean_bps']:+.2f} bps | win rate: {metrics['win_rate_pct']:.1f}%")
    print(f"total: {metrics['total_return_pct']:+.1f}% | CAGR: {metrics['cagr_pct']:+.1f}%")
    print(f"ann vol: {metrics['ann_vol_pct']:.1f}% | Sharpe: {metrics['sharpe']:.2f} | max DD: {metrics['max_drawdown_pct']:.1f}%")
    if not np.isnan(metrics.get("benchmark_excess_pct", np.nan)):
        print(f"excess vs benchmark close-to-close: {metrics['benchmark_excess_pct']:+.1f}%")
    print("\nCaveats: close-to-next-open only; no hidden selection rule inferred yet; no intraday fill/slippage model beyond cost_bps.")
    print("This script is research tooling, not investment advice.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest a fixed close-to-next-open ETF basket.")
    ap.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated ETF basket.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--provider", choices=["auto", "polygon", "yfinance"], default="auto")
    ap.add_argument("--cost-bps", type=float, default=1.0, help="Round-trip cost per basket trade.")
    ap.add_argument("--benchmark", default="SPY", help="Close-to-close benchmark ticker, if loaded.")
    args = ap.parse_args(argv)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    all_tickers = sorted(set(tickers + ([args.benchmark.upper()] if args.benchmark else [])))
    bars, provider = load_daily_ohlc(all_tickers, start=args.start, end=args.end, provider=args.provider)
    overnight = overnight_returns({t: bars[t] for t in tickers if t in bars})
    res = backtest_static_basket(overnight, cost_bps=args.cost_bps)
    bench = benchmark_close_to_close(bars, args.benchmark.upper()) if args.benchmark else None
    metrics = performance_metrics(res, benchmark=bench)
    print_report(res, metrics, title=f"overnight basket [{provider}]")


if __name__ == "__main__":
    main()
