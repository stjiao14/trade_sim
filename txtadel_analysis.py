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
import re
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


def print_audit(orders, daily, returns=None):
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
        print("\nCapped inverse-vol weight fit:")
        print(fit_inverse_vol_weighting(orders, returns).round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Audit Txtadel-style posted overnight ETF signals.")
    ap.add_argument("pdf", nargs="?", help="Path to Txtadel PDF export")
    args = ap.parse_args(argv)
    if not args.pdf:
        raise SystemExit("Pass a Txtadel PDF path.")
    orders, daily = parse_txtadel_pdf(Path(args.pdf))
    print_audit(orders, daily)


if __name__ == "__main__":
    main()
