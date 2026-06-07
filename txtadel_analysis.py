"""Utilities for auditing Txtadel-style overnight ETF basket signals.

The PDF/website exposes posted orders, not the full signal-generation rule.
This module therefore focuses on verifiable claims:

1. Parse posted date/ticker/weight/buy-close/sell-open/gain rows.
2. Recompute weighted overnight return from the posted rows.
3. Test whether posted weights look like capped inverse-vol/risk-parity weights.

It intentionally does not infer a black-box alpha from a short sample.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


DATE_RE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})\s+([0-9:]+)")
ROW_RE = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9]{1,7})\s+"
    r"(?P<weight>[0-9]+(?:\.[0-9]+)?)%\s+"
    r"(?P<buy>\$[0-9,.]+|Pending)\s+"
    r"(?P<sell>\$[0-9,.]+|Pending)\s+"
    r"(?P<gain>[+\-]?[0-9]+(?:\.[0-9]+)?%|Pending)\s*$"
)
TOTAL_RE = re.compile(r"^Total Return:\s*(?P<total>[+\-]?[0-9]+(?:\.[0-9]+)?%|Pending)")


def _parse_money(value):
    if value == "Pending":
        return np.nan
    return float(value.replace("$", "").replace(",", ""))


def _parse_pct(value):
    if value == "Pending":
        return np.nan
    return float(value.replace("%", ""))


def parse_txtadel_text(text):
    """Parse extracted Txtadel PDF text into (orders, daily) DataFrames.

    orders columns:
        date, timestamp, ticker, weight_pct, buy_close, sell_open, gain_pct

    daily columns:
        date, timestamp, posted_total_pct, n_orders
    """
    orders = []
    daily = []
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = DATE_RE.match(line)
        if m:
            current = dict(date=pd.to_datetime(m.group(1)).date(), timestamp=m.group(2))
            continue
        if current is None:
            continue
        m = ROW_RE.match(line)
        if m:
            orders.append({
                **current,
                "ticker": m.group("ticker"),
                "weight_pct": float(m.group("weight")),
                "buy_close": _parse_money(m.group("buy")),
                "sell_open": _parse_money(m.group("sell")),
                "gain_pct": _parse_pct(m.group("gain")),
            })
            continue
        m = TOTAL_RE.match(line)
        if m:
            daily.append({
                **current,
                "posted_total_pct": _parse_pct(m.group("total")),
            })
    odf = pd.DataFrame(orders)
    ddf = pd.DataFrame(daily)
    if not odf.empty:
        counts = odf.groupby("date")["ticker"].size().rename("n_orders")
        if ddf.empty:
            ddf = counts.reset_index()
            ddf["posted_total_pct"] = np.nan
            ddf["timestamp"] = ""
        else:
            ddf = ddf.merge(counts, on="date", how="left")
            ddf["n_orders"] = ddf["n_orders"].fillna(0).astype(int)
    return odf, ddf


def parse_txtadel_pdf(path):
    """Extract text from a PDF and parse Txtadel posted orders."""
    import pypdf

    reader = pypdf.PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return parse_txtadel_text(text)


def _local_config_value(name, default=None):
    """Read optional local config without requiring config_local.py."""
    try:
        import config_local as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


def recompute_daily_returns(orders):
    """Recompute weighted overnight return from posted row-level gains."""
    if orders.empty:
        return pd.DataFrame(columns=["date", "calc_total_pct", "weight_sum_pct", "n_filled"])
    filled = orders.dropna(subset=["gain_pct"]).copy()
    if filled.empty:
        return pd.DataFrame(columns=["date", "calc_total_pct", "weight_sum_pct", "n_filled"])
    filled["weighted"] = filled["weight_pct"] * filled["gain_pct"] / 100.0
    out = (filled.groupby("date")
           .agg(calc_total_pct=("weighted", "sum"),
                weight_sum_pct=("weight_pct", "sum"),
                n_filled=("ticker", "size"))
           .reset_index())
    return out


def compare_posted_vs_recomputed(orders, daily):
    """Compare posted total return with recomputed weighted return."""
    calc = recompute_daily_returns(orders)
    if daily.empty:
        return calc
    out = daily.merge(calc, on="date", how="left")
    out["diff_pct"] = out["calc_total_pct"] - out["posted_total_pct"]
    return out


def required_history_window(orders, max_lookback=120, buffer_days=80):
    """Return a calendar start/end window wide enough for pre-signal return history."""
    if orders.empty:
        raise ValueError("orders is empty")
    d = pd.to_datetime(orders["date"])
    start = (d.min() - pd.Timedelta(days=int(max_lookback * 2 + buffer_days))).date().isoformat()
    end = (d.max() + pd.Timedelta(days=1)).date().isoformat()
    return start, end


def returns_from_closes(closes):
    """Convert a close-price panel into daily close-to-close returns."""
    if closes.empty:
        return pd.DataFrame()
    out = closes.sort_index().pct_change().dropna(how="all").fillna(0.0)
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out


def load_daily_closes_yfinance(tickers, start, end):
    """Load adjusted daily closes via yfinance."""
    import yfinance as yf

    raw = yf.download(list(tickers), start=start, end=end, interval="1d",
                      progress=False, auto_adjust=True)["Close"]
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(list(tickers)[0])
    raw = raw.dropna(axis=1, how="all")
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    return raw


def _polygon_daily_close(ticker, start, end, api_key):
    params = urllib.parse.urlencode(dict(adjusted="true", sort="asc", limit=50000, apiKey=api_key))
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    rows = data.get("results", [])
    if not rows:
        return pd.Series(dtype=float, name=ticker)
    df = pd.DataFrame(rows)
    idx = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return pd.Series(df["c"].astype(float).values, index=idx, name=ticker)


def load_daily_closes_polygon(tickers, start, end, api_key=None):
    """Load adjusted daily closes via Polygon/Massive."""
    key = api_key or os.getenv("POLYGON_API_KEY") or _local_config_value("POLYGON_API_KEY")
    if not key:
        raise ValueError("POLYGON_API_KEY is not configured")
    series = [_polygon_daily_close(t, start, end, key) for t in tickers]
    return pd.concat(series, axis=1).dropna(axis=1, how="all")


def load_daily_returns_for_orders(orders, lookbacks=(20, 40, 60, 120), provider="auto", api_key=None):
    """Load daily returns needed to fit posted weights.

    provider='auto' prefers Polygon/Massive when a key is configured, then falls back
    to yfinance.
    """
    if orders.empty:
        return pd.DataFrame(), "none"
    tickers = sorted(orders["ticker"].dropna().unique())
    start, end = required_history_window(orders, max_lookback=max(lookbacks))
    provider = provider.lower()
    if provider not in {"auto", "polygon", "yfinance"}:
        raise ValueError("provider must be auto, polygon, or yfinance")
    if provider in {"auto", "polygon"}:
        try:
            closes = load_daily_closes_polygon(tickers, start, end, api_key=api_key)
            if not closes.empty:
                return returns_from_closes(closes), "polygon"
        except Exception:
            if provider == "polygon":
                raise
    closes = load_daily_closes_yfinance(tickers, start, end)
    return returns_from_closes(closes), "yfinance"


def capped_normalize(raw_scores, cap=0.35):
    """Normalize positive scores to weights with an iterative max-weight cap."""
    s = pd.Series(raw_scores, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if s.empty:
        return pd.Series(dtype=float)
    remaining = list(s.index)
    weights = pd.Series(0.0, index=s.index, dtype=float)
    remaining_mass = 1.0
    while remaining:
        vals = s.loc[remaining]
        provisional = vals / vals.sum() * remaining_mass
        capped = provisional[provisional > cap]
        if capped.empty:
            weights.loc[remaining] = provisional
            break
        for idx in capped.index:
            weights.loc[idx] = cap
            remaining_mass -= cap
            remaining.remove(idx)
        if remaining_mass <= 1e-12:
            break
    total = weights.sum()
    return weights / total if total > 0 else weights


def inverse_vol_weights(returns, tickers, asof_date, lookback=60, cap=0.35):
    """Compute capped inverse-vol weights using returns strictly before asof_date."""
    asof = pd.to_datetime(asof_date)
    hist = returns.loc[returns.index < asof, list(tickers)].tail(lookback)
    vol = hist.std(ddof=1).replace(0, np.nan)
    return capped_normalize(1.0 / vol, cap=cap)


def fit_inverse_vol_weighting(orders, returns, lookbacks=(20, 40, 60, 120), cap=0.35):
    """Score how closely posted weights match capped inverse-vol weights."""
    rows = []
    clean = orders.dropna(subset=["weight_pct"]).copy()
    if clean.empty:
        return pd.DataFrame(columns=["lookback", "n_dates", "mae_pct", "rmse_pct", "corr"])
    clean["date"] = pd.to_datetime(clean["date"])
    for lookback in lookbacks:
        obs = []
        pred = []
        for date, g in clean.groupby("date"):
            tickers = list(g["ticker"])
            if not set(tickers) <= set(returns.columns):
                continue
            w = inverse_vol_weights(returns, tickers, date, lookback=lookback, cap=cap)
            if w.empty:
                continue
            posted = g.set_index("ticker")["weight_pct"] / 100.0
            common = posted.index.intersection(w.index)
            obs.extend(posted.loc[common].values)
            pred.extend(w.loc[common].values)
        if not obs:
            rows.append(dict(lookback=lookback, n_dates=0, mae_pct=np.nan, rmse_pct=np.nan, corr=np.nan))
            continue
        obs = np.asarray(obs)
        pred = np.asarray(pred)
        err = pred - obs
        corr = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 and np.std(pred) > 0 and np.std(obs) > 0 else np.nan
        rows.append(dict(
            lookback=int(lookback),
            n_dates=int(clean["date"].nunique()),
            mae_pct=float(np.mean(np.abs(err)) * 100),
            rmse_pct=float(np.sqrt(np.mean(err ** 2)) * 100),
            corr=float(corr) if np.isfinite(corr) else np.nan,
        ))
    return pd.DataFrame(rows)


def _rule_scores(hist, rule):
    """Score tickers from a trailing return panel. Higher score means more likely selected."""
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
    raise ValueError(f"unknown rule: {rule}")


def candidate_selection(returns, asof_date, lookback=60, rule="mean", top_n=5, universe=None):
    """Select candidate tickers using only returns strictly before asof_date."""
    asof = pd.to_datetime(asof_date)
    cols = list(universe) if universe is not None else list(returns.columns)
    cols = [c for c in cols if c in returns.columns]
    if not cols:
        return []
    hist = returns.loc[returns.index < asof, cols].tail(int(lookback))
    sc = _rule_scores(hist, rule).replace([np.inf, -np.inf], np.nan).dropna()
    if sc.empty:
        return []
    return list(sc.sort_values(ascending=False).head(int(top_n)).index)


def score_candidate_selection_rules(orders, returns, lookbacks=(20, 40, 60, 120),
                                    rules=("mean", "tstat", "momentum", "reversal", "low_vol"),
                                    top_n=5, universe=None):
    """Compare simple candidate selection rules with posted Txtadel baskets.

    This is a reverse-engineering aid, not proof of the hidden rule. It reports
    how many posted tickers each transparent rule would have selected using only
    history before the posted date.
    """
    if orders.empty or returns.empty:
        return pd.DataFrame(columns=[
            "rule", "lookback", "n_dates", "avg_overlap", "avg_overlap_pct",
            "exact_match_days", "freq_corr"
        ])
    clean = orders.dropna(subset=["ticker"]).copy()
    clean["date"] = pd.to_datetime(clean["date"])
    if universe is None:
        universe = sorted(set(clean["ticker"]).union(set(returns.columns)))
    rows = []
    for rule in rules:
        for lookback in lookbacks:
            overlaps = []
            posted_sizes = []
            exact = 0
            pred_counts = pd.Series(dtype=float)
            posted_counts = pd.Series(dtype=float)
            n_dates = 0
            for date, g in clean.groupby("date"):
                posted = set(g["ticker"].dropna())
                if not posted:
                    continue
                pred = set(candidate_selection(returns, date, lookback=lookback,
                                               rule=rule, top_n=top_n, universe=universe))
                if not pred:
                    continue
                n_dates += 1
                hit = len(posted & pred)
                overlaps.append(hit)
                posted_sizes.append(len(posted))
                exact += int(posted == pred)
                posted_counts = posted_counts.add(pd.Series(1.0, index=list(posted)), fill_value=0.0)
                pred_counts = pred_counts.add(pd.Series(1.0, index=list(pred)), fill_value=0.0)
            if not overlaps:
                rows.append(dict(rule=rule, lookback=int(lookback), n_dates=0,
                                 avg_overlap=np.nan, avg_overlap_pct=np.nan,
                                 exact_match_days=0, freq_corr=np.nan))
                continue
            all_names = sorted(set(posted_counts.index).union(set(pred_counts.index)))
            obs = posted_counts.reindex(all_names, fill_value=0.0).values
            prd = pred_counts.reindex(all_names, fill_value=0.0).values
            corr = np.corrcoef(obs, prd)[0, 1] if len(all_names) > 1 and np.std(obs) > 0 and np.std(prd) > 0 else np.nan
            rows.append(dict(
                rule=rule,
                lookback=int(lookback),
                n_dates=int(n_dates),
                avg_overlap=float(np.mean(overlaps)),
                avg_overlap_pct=float(np.mean([x / max(sz, 1) for x, sz in zip(overlaps, posted_sizes)]) * 100),
                exact_match_days=int(exact),
                freq_corr=float(corr) if np.isfinite(corr) else np.nan,
            ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["avg_overlap", "freq_corr"], ascending=[False, False]).reset_index(drop=True)
    return out


def print_audit(orders, daily, returns=None, lookbacks=(20, 40, 60, 120), cap=0.35,
                provider_used=None, score_selection=False, top_n=5):
    """Print a compact Txtadel posted-signal audit."""
    print("== Txtadel posted-order audit ==")
    print(f"dates: {orders['date'].nunique() if not orders.empty else 0} | rows: {len(orders)}")
    print("\nPosted vs recomputed daily return:")
    cmp = compare_posted_vs_recomputed(orders, daily)
    if cmp.empty:
        print("No completed rows.")
    else:
        cols = [c for c in ["date", "posted_total_pct", "calc_total_pct", "diff_pct", "weight_sum_pct", "n_filled"] if c in cmp]
        print(cmp[cols].round(4).to_string(index=False))
    print("\nTicker frequency:")
    if not orders.empty:
        print(orders["ticker"].value_counts().to_string())
    if returns is not None:
        label = f" [{provider_used}]" if provider_used else ""
        print(f"\nCapped inverse-vol weight fit{label}:")
        print(fit_inverse_vol_weighting(orders, returns, lookbacks=lookbacks, cap=cap).round(4).to_string(index=False))
        if score_selection:
            print(f"\nCandidate selection-rule overlap{label}:")
            sel = score_candidate_selection_rules(orders, returns, lookbacks=lookbacks, top_n=top_n)
            print(sel.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit Txtadel-style posted overnight ETF signals.")
    ap.add_argument("pdf", nargs="?", help="Path to Txtadel PDF export")
    ap.add_argument("--fit-weights", action="store_true",
                    help="Load ETF return history and fit capped inverse-vol weights.")
    ap.add_argument("--score-selection", action="store_true",
                    help="Load ETF return history and score simple top-N selection rules against posted tickers.")
    ap.add_argument("--provider", choices=["auto", "polygon", "yfinance"], default="auto",
                    help="Daily data provider for --fit-weights. auto prefers Polygon/Massive when configured.")
    ap.add_argument("--lookbacks", default="20,40,60,120",
                    help="Comma-separated lookback windows for inverse-vol fit.")
    ap.add_argument("--cap", type=float, default=0.35,
                    help="Max single-ticker weight for capped inverse-vol fit.")
    ap.add_argument("--top-n", type=int, default=5,
                    help="Number of tickers selected by each candidate rule.")
    args = ap.parse_args(argv)
    if not args.pdf:
        raise SystemExit("Pass a Txtadel PDF path.")
    orders, daily = parse_txtadel_pdf(Path(args.pdf))
    lookbacks = tuple(int(x.strip()) for x in args.lookbacks.split(",") if x.strip())
    returns = None
    provider_used = None
    if args.fit_weights or args.score_selection:
        returns, provider_used = load_daily_returns_for_orders(orders, lookbacks=lookbacks, provider=args.provider)
    print_audit(orders, daily, returns=returns, lookbacks=lookbacks, cap=args.cap,
                provider_used=provider_used, score_selection=args.score_selection, top_n=args.top_n)


if __name__ == "__main__":
    main()
