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
EXPANDED_ETF_UNIVERSE = [
    "SPY", "QQQ", "DIA", "IWM",
    "XLK", "XLF", "XLY", "XLP", "XLU", "XLE", "XLI", "XLV", "XLB", "XLRE", "XLC",
    "SMH", "IGV", "IYT", "KRE", "XBI",
    "MTUM", "QUAL", "USMV", "VLUE", "IWD", "IWF", "RSP",
    "EFA", "EEM", "EMXC", "EWJ", "FEZ",
    "TLT", "IEF", "SHY", "LQD", "HYG",
    "GLD", "SLV", "USO", "DBC",
    "CGDV", "AVUV",
]
TRADING_DAYS = 252
DEFAULT_VIX_TICKER = "^VIX"
DEFAULT_RISK_ON = "XLY"
DEFAULT_RISK_OFF = "XLP"
DEFAULT_ETF_COST_BPS = {
    "SPY": 0.5, "QQQ": 0.5, "DIA": 0.7, "IWM": 0.8,
    "XLK": 0.8, "XLF": 0.8, "XLY": 0.8, "XLP": 0.8, "XLU": 0.8,
    "XLE": 0.8, "XLI": 0.8, "XLV": 0.8, "XLB": 0.9, "XLRE": 1.0, "XLC": 1.0,
    "SMH": 1.0, "IGV": 1.2, "IYT": 1.5, "KRE": 1.5, "XBI": 1.8,
    "MTUM": 1.2, "QUAL": 1.0, "USMV": 1.0, "VLUE": 1.2, "IWD": 1.0, "IWF": 1.0, "RSP": 1.0,
    "EFA": 1.0, "EEM": 1.8, "EMXC": 2.5, "EWJ": 1.2, "FEZ": 1.5,
    "TLT": 1.0, "IEF": 0.8, "SHY": 0.8, "LQD": 1.5, "HYG": 1.8,
    "GLD": 0.8, "SLV": 1.5, "USO": 2.0, "DBC": 2.0,
    "CGDV": 2.5, "AVUV": 2.5,
}


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


def parse_tickers(value):
    """Parse comma-separated tickers or the shortcut 'expanded'."""
    if isinstance(value, str) and value.strip().lower() == "expanded":
        return list(dict.fromkeys(EXPANDED_ETF_UNIVERSE))
    if isinstance(value, str):
        return [t.strip().upper() for t in value.split(",") if t.strip()]
    return [str(t).upper() for t in value]


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


def close_to_close_returns(bars):
    """Return close-to-close daily returns for scoring/benchmarking."""
    series = {}
    for ticker, df in bars.items():
        b = _clean_ohlc_frame(df)
        series[ticker] = b["Close"].pct_change().dropna()
    return pd.DataFrame(series).sort_index()


def _score_history(hist, rule):
    """Score a trailing return panel for transparent ETF selection rules."""
    hist = hist.dropna(axis=1, how="all")
    if hist.empty:
        return pd.Series(dtype=float)
    if rule == "mean":
        return hist.mean()
    if rule == "tstat":
        vol = hist.std(ddof=1).replace(0, np.nan)
        return hist.mean() / vol * np.sqrt(hist.count())
    if rule == "momentum":
        return (1.0 + hist.fillna(0.0)).prod() - 1.0
    if rule == "reversal":
        return -((1.0 + hist.fillna(0.0)).prod() - 1.0)
    if rule == "low_vol":
        return 1.0 / hist.std(ddof=1).replace(0, np.nan)
    raise ValueError("rule must be mean, tstat, momentum, reversal, or low_vol")


def select_top_n(signal_returns, asof_date, rule="reversal", lookback=60, top_n=5, universe=None):
    """Select top-N ETFs using only signal returns strictly before asof_date."""
    asof = pd.to_datetime(asof_date).tz_localize(None).normalize()
    cols = list(universe) if universe is not None else list(signal_returns.columns)
    cols = [c for c in cols if c in signal_returns.columns]
    if not cols:
        return pd.Series(dtype=float)
    hist = signal_returns.loc[signal_returns.index < asof, cols].tail(int(lookback))
    if len(hist) < int(lookback):
        return pd.Series(dtype=float)
    sc = _score_history(hist, rule).replace([np.inf, -np.inf], np.nan).dropna()
    return sc.sort_values(ascending=False).head(int(top_n))


def realized_cost_bps(weights, cost_bps=1.0, cost_by_ticker=None):
    """Weighted round-trip cost for a basket."""
    w = pd.Series(weights, dtype=float)
    if w.empty:
        return 0.0
    if cost_by_ticker:
        c = pd.Series({t: float(cost_by_ticker.get(t, cost_bps)) for t in w.index})
        return float((w * c.reindex(w.index)).sum())
    return float(cost_bps)


def basket_weights(selected, signal_returns, asof_date, weighting="equal", cap=0.35, lookback=60):
    """Build basket weights from selected tickers."""
    tickers = list(selected.index) if isinstance(selected, pd.Series) else list(selected)
    if not tickers:
        return pd.Series(dtype=float)
    if weighting == "equal":
        return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
    if weighting == "score":
        sc = pd.Series(selected, dtype=float)
        sc = sc - sc.min()
        if sc.sum() <= 0:
            return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
        return (sc / sc.sum()).reindex(tickers)
    if weighting == "inv_vol":
        asof = pd.to_datetime(asof_date).tz_localize(None).normalize()
        hist = signal_returns.loc[signal_returns.index < asof, tickers].tail(int(lookback))
        vol = hist.std(ddof=1).replace(0, np.nan)
        raw = 1.0 / vol
        raw = raw.replace([np.inf, -np.inf], np.nan).dropna()
        if raw.empty:
            return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
        return cap_weights(raw / raw.sum(), cap=cap).reindex(tickers).fillna(0.0)
    raise ValueError("weighting must be equal, score, or inv_vol")


def cap_weights(weights, cap=0.35):
    """Normalize positive weights with an iterative max-weight cap."""
    s = pd.Series(weights, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty:
        return pd.Series(dtype=float)
    remaining = list(s.index)
    out = pd.Series(0.0, index=s.index, dtype=float)
    mass = 1.0
    while remaining:
        vals = s.loc[remaining]
        w = vals / vals.sum() * mass
        over = w[w > cap]
        if over.empty:
            out.loc[remaining] = w
            break
        for idx in over.index:
            out.loc[idx] = cap
            remaining.remove(idx)
            mass -= cap
        if mass <= 1e-12:
            break
    return out / out.sum()


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


def backtest_selection_rule(overnight, signal_returns, rule="reversal", lookback=60, top_n=5,
                            universe=None, weighting="equal", cap=0.35, cost_bps=1.0,
                            cost_by_ticker=None, regime=None, vix_max=30.0,
                            macro_min=-0.20, decision_lag=1):
    """Walk-forward overnight backtest for a transparent ETF selection rule."""
    rows = []
    rg = None
    if regime is not None:
        rg = regime[["vix", "macro_roc"]].shift(int(decision_lag))
    for date, row in overnight.sort_index().iterrows():
        sc = select_top_n(signal_returns, date, rule=rule, lookback=lookback,
                          top_n=top_n, universe=universe)
        if sc.empty:
            continue
        if rg is not None:
            if date not in rg.index:
                continue
            state = rg.loc[date]
            if pd.isna(state["vix"]) or pd.isna(state["macro_roc"]):
                continue
            if not (state["vix"] <= vix_max and state["macro_roc"] >= macro_min):
                continue
        available = row.reindex(sc.index).dropna()
        if available.empty:
            continue
        sc = sc.reindex(available.index).dropna()
        w = basket_weights(sc, signal_returns, date, weighting=weighting, cap=cap, lookback=lookback)
        w = w.reindex(available.index).dropna()
        if w.empty or w.sum() <= 0:
            continue
        w = w / w.sum()
        gross = float((available.loc[w.index] * w).sum())
        cbps = realized_cost_bps(w, cost_bps=cost_bps, cost_by_ticker=cost_by_ticker)
        rows.append(dict(date=date, gross=gross, net=gross - cbps / 1e4,
                         n_assets=int(len(w)), picks=",".join(w.index),
                         rule=rule, lookback=int(lookback), top_n=int(top_n),
                         weighting=weighting, cost_bps=float(cbps)))
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


def summarize_rule_grid(overnight, signal_returns, rules=("reversal", "mean", "tstat", "momentum", "low_vol"),
                        lookbacks=(20, 40, 60, 120), top_ns=(5,), universe=None,
                        weighting="equal", cap=0.35, cost_bps=1.0, cost_by_ticker=None,
                        regime=None, vix_max=30.0, macro_min=-0.20, decision_lag=1,
                        benchmark=None):
    """Run a compact parameter grid and return one metrics row per rule."""
    rows = []
    for rule in rules:
        for lookback in lookbacks:
            for top_n in top_ns:
                res = backtest_selection_rule(
                    overnight, signal_returns, rule=rule, lookback=lookback,
                    top_n=top_n, universe=universe, weighting=weighting, cap=cap,
                    cost_bps=cost_bps, cost_by_ticker=cost_by_ticker, regime=regime,
                    vix_max=vix_max, macro_min=macro_min, decision_lag=decision_lag,
                )
                m = performance_metrics(res, benchmark=benchmark)
                rows.append(dict(rule=rule, lookback=int(lookback), top_n=int(top_n),
                                 weighting=weighting, **m))
    return pd.DataFrame(rows).sort_values(["sharpe", "cagr_pct"], ascending=[False, False]).reset_index(drop=True)


def period_splits(index, freq="Y"):
    """Return {period_label: DatetimeIndex} for calendar OOS segments."""
    idx = pd.DatetimeIndex(pd.to_datetime(index)).tz_localize(None).normalize()
    periods = idx.to_series(index=idx).dt.to_period(freq)
    return {str(p): idx[periods.values == p] for p in sorted(periods.unique())}


def walk_forward_oos_grid(overnight, signal_returns, grid, freq="Y", benchmark=None,
                          universe=None, cost_by_ticker=None, regime=None,
                          vix_max=30.0, macro_min=-0.20, decision_lag=1):
    """Evaluate fixed candidate parameters by calendar OOS segment.

    The parameters are not refit inside the segment; each row asks whether that
    transparent rule remains stable across calendar regimes.
    """
    rows = []
    splits = period_splits(overnight.index, freq=freq)
    for label, dates in splits.items():
        ov = overnight.reindex(dates).dropna(how="all")
        if ov.empty:
            continue
        bench = benchmark.reindex(dates).dropna() if benchmark is not None else None
        reg = regime.reindex(dates.union(regime.index)).sort_index() if regime is not None else None
        for cfg in grid:
            res = backtest_selection_rule(
                ov, signal_returns,
                rule=cfg.get("rule", "reversal"),
                lookback=int(cfg.get("lookback", 60)),
                top_n=int(cfg.get("top_n", 5)),
                universe=universe,
                weighting=cfg.get("weighting", "equal"),
                cap=float(cfg.get("cap", 0.35)),
                cost_bps=float(cfg.get("cost_bps", 1.0)),
                cost_by_ticker=cost_by_ticker,
                regime=reg,
                vix_max=vix_max,
                macro_min=macro_min,
                decision_lag=decision_lag,
            )
            m = performance_metrics(res, benchmark=bench)
            rows.append(dict(period=label, **cfg, **m))
    return pd.DataFrame(rows)


def next_selection_plan(bars, rule="reversal", lookback=60, top_n=5, universe=None,
                        weighting="equal", cap=0.35, notional=1000.0,
                        signal_kind="overnight"):
    """Build a next-session shadow plan from the latest available history."""
    if signal_kind == "overnight":
        sig = overnight_returns(bars)
    elif signal_kind == "close":
        sig = close_to_close_returns(bars)
    else:
        raise ValueError("signal_kind must be overnight or close")
    if sig.empty:
        return pd.DataFrame()
    asof = sig.index.max() + pd.Timedelta(days=1)
    sc = select_top_n(sig, asof, rule=rule, lookback=lookback, top_n=top_n, universe=universe)
    if sc.empty:
        return pd.DataFrame()
    w = basket_weights(sc, sig, asof, weighting=weighting, cap=cap, lookback=lookback)
    rows = []
    for ticker, weight in w.items():
        rows.append(dict(ticker=ticker, weight=float(weight),
                         notional=float(weight * notional), score=float(sc.get(ticker, np.nan)),
                         rule=rule, lookback=int(lookback), top_n=int(top_n),
                         weighting=weighting, signal_kind=signal_kind,
                         history_end=sig.index.max().date()))
    return pd.DataFrame(rows)


def to_signal_lab_panel(returns, slot=0):
    """Convert daily ETF returns to signal_lab's date/slot/ticker/ret panel."""
    frames = []
    for ticker in returns.columns:
        s = returns[ticker].dropna()
        frames.append(pd.DataFrame({
            "date": pd.to_datetime(s.index).date,
            "slot": int(slot),
            "ticker": ticker,
            "ret": s.values,
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["date", "slot", "ticker", "ret"])


def overnight_signal_fn(rule):
    """Return a signal_lab-compatible function for an overnight rule."""
    def _fn(hist):
        return _score_history(hist, rule)
    _fn.__name__ = f"overnight_{rule}_signal"
    return _fn


def benchmark_close_to_close(bars, ticker="SPY"):
    """Compute close-to-close benchmark returns from a bars dict."""
    if ticker not in bars:
        return pd.Series(dtype=float, name=ticker)
    b = _clean_ohlc_frame(bars[ticker])
    return b["Close"].pct_change().dropna().rename(ticker)


def regime_panel(bars, vix_ticker=DEFAULT_VIX_TICKER, risk_on=DEFAULT_RISK_ON,
                 risk_off=DEFAULT_RISK_OFF, macro_lookback=200):
    """Build the two-axis risk-control panel.

    macro_roc follows the described rule:
        ((XLY/XLP)_t - (XLY/XLP)_{t-lookback}) / (XLY/XLP)_t

    Higher macro_roc means the risk-on/risk-off ratio is healthier versus its
    long lookback. Lower values are treated as macro stress.
    """
    missing = [t for t in (vix_ticker, risk_on, risk_off) if t not in bars]
    if missing:
        raise ValueError(f"missing regime tickers: {missing}")
    vix = _clean_ohlc_frame(bars[vix_ticker])["Close"].rename("vix")
    ro = _clean_ohlc_frame(bars[risk_on])["Close"]
    rf = _clean_ohlc_frame(bars[risk_off])["Close"]
    ratio = (ro / rf).rename("risk_ratio")
    macro_roc = ((ratio - ratio.shift(macro_lookback)) / ratio).rename("macro_roc")
    return pd.concat([vix, ratio, macro_roc], axis=1).dropna(how="any")


def apply_regime_gate(res, regime, vix_max=30.0, macro_min=-0.20, decision_lag=1):
    """Filter trades by VIX and macro-ratio conditions.

    decision_lag=1 uses the previous session's completed regime data for today's
    close order. decision_lag=0 reproduces a same-day-close rule, but that is
    only appropriate if the implementation truly has the needed inputs before
    sending the MOC order.
    """
    if res.empty:
        return res.copy()
    rg = regime[["vix", "macro_roc"]].shift(int(decision_lag))
    out = res.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    joined = out.join(rg, on="date")
    joined["trade_allowed"] = (joined["vix"] <= vix_max) & (joined["macro_roc"] >= macro_min)
    return joined[joined["trade_allowed"]].reset_index(drop=True)


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


def print_gate_comparison(base_res, gated_res, bench=None, title="VIX + macro gate"):
    """Print before/after performance for a risk gate."""
    base_m = performance_metrics(base_res, benchmark=bench)
    gate_m = performance_metrics(gated_res, benchmark=bench)
    print(f"\n== {title} comparison ==")
    print("version        days   mean bps   win%   CAGR%   Sharpe   maxDD%")
    for name, m in [("ungated", base_m), ("gated", gate_m)]:
        print(f"{name:<12} {m['n_days']:>5} {m['mean_bps']:>10.2f} {m['win_rate_pct']:>6.1f} "
              f"{m['cagr_pct']:>7.1f} {m['sharpe']:>8.2f} {m['max_drawdown_pct']:>8.1f}")
    if len(base_res):
        kept = len(gated_res) / len(base_res) * 100
        print(f"gate kept {kept:.1f}% of overnight trades")


def _parse_ints(value):
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


def _parse_strings(value):
    return tuple(x.strip() for x in str(value).split(",") if x.strip())


def _print_grid(df, title, n=20):
    print(f"\n== {title} ==")
    if df.empty:
        print("No rows.")
        return
    cols = [c for c in ["period", "rule", "lookback", "top_n", "weighting", "n_days",
                        "mean_bps", "win_rate_pct", "cagr_pct", "sharpe",
                        "max_drawdown_pct", "benchmark_excess_pct"] if c in df.columns]
    print(df[cols].head(n).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Backtest a fixed close-to-next-open ETF basket.")
    ap.add_argument("--tickers", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated ETF basket.")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--provider", choices=["auto", "polygon", "yfinance"], default="auto")
    ap.add_argument("--cost-bps", type=float, default=1.0, help="Round-trip cost per basket trade.")
    ap.add_argument("--cost-model", choices=["flat", "etf"], default="flat",
                    help="flat uses --cost-bps; etf uses a rough per-ETF round-trip cost table.")
    ap.add_argument("--benchmark", default="SPY", help="Close-to-close benchmark ticker, if loaded.")
    ap.add_argument("--mode", choices=["static", "rule", "grid", "oos", "shadow-plan"], default="static",
                    help="static=fixed basket, rule=one dynamic rule, grid=parameter table, oos=calendar segments, shadow-plan=next basket.")
    ap.add_argument("--rule", choices=["mean", "tstat", "momentum", "reversal", "low_vol"], default="reversal")
    ap.add_argument("--rules", default="reversal,mean,tstat,momentum,low_vol")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--lookbacks", default="20,40,60,120")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--top-ns", default="5")
    ap.add_argument("--weighting", choices=["equal", "score", "inv_vol"], default="equal")
    ap.add_argument("--cap", type=float, default=0.35)
    ap.add_argument("--signal-kind", choices=["overnight", "close"], default="overnight",
                    help="Return panel used for rule scoring.")
    ap.add_argument("--oos-freq", default="Y", help="Calendar split frequency for --mode oos, e.g. Y or Q.")
    ap.add_argument("--notional", type=float, default=10_000.0, help="Notional for --mode shadow-plan.")
    ap.add_argument("--gate", choices=["none", "vix-macro"], default="none",
                    help="Optional risk gate applied before the close-to-open trade.")
    ap.add_argument("--vix-ticker", default=DEFAULT_VIX_TICKER)
    ap.add_argument("--risk-on", default=DEFAULT_RISK_ON, help="Risk-on ETF for macro ratio numerator.")
    ap.add_argument("--risk-off", default=DEFAULT_RISK_OFF, help="Risk-off ETF for macro ratio denominator.")
    ap.add_argument("--macro-lookback", type=int, default=200)
    ap.add_argument("--vix-max", type=float, default=30.0)
    ap.add_argument("--macro-min", type=float, default=-0.20)
    ap.add_argument("--decision-lag", type=int, default=1,
                    help="Regime rows to lag before trading. 1 avoids same-close lookahead.")
    args = ap.parse_args(argv)

    tickers = parse_tickers(args.tickers)
    extras = [args.benchmark.upper()] if args.benchmark else []
    if args.gate == "vix-macro":
        extras += [args.vix_ticker.upper(), args.risk_on.upper(), args.risk_off.upper()]
    all_tickers = sorted(set(tickers + extras))
    bars, provider = load_daily_ohlc(all_tickers, start=args.start, end=args.end, provider=args.provider)
    overnight = overnight_returns({t: bars[t] for t in tickers if t in bars})
    signal_returns = overnight if args.signal_kind == "overnight" else close_to_close_returns({t: bars[t] for t in tickers if t in bars})
    bench = benchmark_close_to_close(bars, args.benchmark.upper()) if args.benchmark else None
    cost_by_ticker = DEFAULT_ETF_COST_BPS if args.cost_model == "etf" else None
    reg = None
    if args.gate == "vix-macro":
        reg = regime_panel(bars, vix_ticker=args.vix_ticker.upper(),
                           risk_on=args.risk_on.upper(), risk_off=args.risk_off.upper(),
                           macro_lookback=args.macro_lookback)
    if args.mode == "shadow-plan":
        plan = next_selection_plan({t: bars[t] for t in tickers if t in bars},
                                   rule=args.rule, lookback=args.lookback, top_n=args.top_n,
                                   universe=tickers, weighting=args.weighting, cap=args.cap,
                                   notional=args.notional, signal_kind=args.signal_kind)
        print("== overnight shadow plan ==")
        print(plan.round(6).to_string(index=False) if not plan.empty else "No plan.")
        print("\nCaveat: this is a shadow plan only; it does not submit broker orders.")
        return
    if args.mode == "grid":
        grid = summarize_rule_grid(
            overnight, signal_returns, rules=_parse_strings(args.rules),
            lookbacks=_parse_ints(args.lookbacks), top_ns=_parse_ints(args.top_ns),
            universe=tickers, weighting=args.weighting, cap=args.cap,
            cost_bps=args.cost_bps, cost_by_ticker=cost_by_ticker, regime=reg,
            vix_max=args.vix_max, macro_min=args.macro_min, decision_lag=args.decision_lag,
            benchmark=bench,
        )
        _print_grid(grid, f"selection-rule grid [{provider}]")
        return
    if args.mode == "oos":
        cfgs = []
        for rule in _parse_strings(args.rules):
            for lb in _parse_ints(args.lookbacks):
                for tn in _parse_ints(args.top_ns):
                    cfgs.append(dict(rule=rule, lookback=int(lb), top_n=int(tn),
                                     weighting=args.weighting, cap=args.cap,
                                     cost_bps=args.cost_bps))
        oos = walk_forward_oos_grid(
            overnight, signal_returns, cfgs, freq=args.oos_freq, benchmark=bench,
            universe=tickers, cost_by_ticker=cost_by_ticker, regime=reg,
            vix_max=args.vix_max, macro_min=args.macro_min, decision_lag=args.decision_lag,
        )
        _print_grid(oos.sort_values(["period", "sharpe"], ascending=[True, False]),
                    f"OOS calendar segments [{provider}]", n=200)
        return
    if args.mode == "rule":
        res = backtest_selection_rule(
            overnight, signal_returns, rule=args.rule, lookback=args.lookback,
            top_n=args.top_n, universe=tickers, weighting=args.weighting,
            cap=args.cap, cost_bps=args.cost_bps, cost_by_ticker=cost_by_ticker,
            regime=reg, vix_max=args.vix_max, macro_min=args.macro_min,
            decision_lag=args.decision_lag,
        )
        final_res = res
    else:
        res = backtest_static_basket(overnight, cost_bps=args.cost_bps)
        final_res = res
        if args.gate == "vix-macro":
            final_res = apply_regime_gate(res, reg, vix_max=args.vix_max,
                                          macro_min=args.macro_min, decision_lag=args.decision_lag)
    metrics = performance_metrics(final_res, benchmark=bench)
    label = f"overnight basket [{provider}]"
    if args.mode == "rule":
        label += f" | rule={args.rule}/{args.lookback}/top{args.top_n}/{args.weighting}"
    if args.gate != "none":
        label += f" | gate={args.gate}"
    print_report(final_res, metrics, title=label)
    if args.gate == "vix-macro" and args.mode == "static":
        print_gate_comparison(res, final_res, bench=bench,
                              title=f"VIX<={args.vix_max:g}, ROC({args.risk_on.upper()}/{args.risk_off.upper()},{args.macro_lookback})>={args.macro_min:g}, lag={args.decision_lag}")


if __name__ == "__main__":
    main()
