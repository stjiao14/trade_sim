"""Daily shadow diary for overnight ETF baskets.

This script does not submit broker orders. It creates an auditable paper diary:

1. plan: choose the next overnight basket and append it to CSV.
2. settle: after the next open is available, compute realized close-to-open P&L.
3. report: print rolling shadow metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import overnight_basket_backtest as ob


PLAN_FILE = "overnight_plans.csv"
SETTLE_FILE = "overnight_settlements.csv"
REPORT_FILE = "overnight_report.csv"


def _append_csv(df, path, key_cols):
    """Append rows, replacing existing rows with the same key."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=key_cols, keep="last")
    else:
        combined = df.copy()
    combined.to_csv(path, index=False)
    return combined


def _read_csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _latest_bars(tickers, start, end=None, provider="auto"):
    bars, provider_used = ob.load_daily_ohlc(tickers, start=start, end=end, provider=provider)
    return bars, provider_used


def create_plan(bars, rule="mean", lookback=40, top_n=5, universe=None,
                weighting="equal", cap=0.35, notional=10_000.0,
                signal_kind="overnight", plan_date=None, allow_pending=True):
    """Create one shadow plan DataFrame with a plan_id."""
    sig = ob.overnight_returns(bars) if signal_kind == "overnight" else ob.close_to_close_returns(bars)
    if sig.empty:
        return pd.DataFrame()
    latest_close = min(
        ob._clean_ohlc_frame(b)["Close"].dropna().index.max()
        for b in bars.values() if not b.empty
    )
    history_end = latest_close if allow_pending else sig.index.max()
    asof = history_end + pd.Timedelta(days=1)
    sc = ob.select_top_n(sig, asof, rule=rule, lookback=lookback, top_n=top_n, universe=universe)
    if sc.empty:
        return pd.DataFrame()
    w = ob.basket_weights(sc, sig, asof, weighting=weighting, cap=cap, lookback=lookback)
    pd_date = pd.to_datetime(plan_date).date() if plan_date else history_end.date()
    plan = pd.DataFrame([
        dict(ticker=ticker, weight=float(weight), notional=float(weight * notional),
             score=float(sc.get(ticker, np.nan)), rule=rule, lookback=int(lookback),
             top_n=int(top_n), weighting=weighting, signal_kind=signal_kind,
             history_end=history_end.date())
        for ticker, weight in w.items()
    ])
    plan["plan_date"] = pd_date.isoformat()
    plan["plan_id"] = f"{pd_date.isoformat()}_{rule}_{lookback}_{top_n}_{weighting}"
    plan["status"] = "planned"
    return plan


def settle_plan(plan, bars, cost_bps=1.0, cost_by_ticker=None):
    """Settle one plan using buy close on history_end and next available open."""
    if plan.empty:
        return pd.DataFrame()
    rows = []
    gross_parts = []
    for row in plan.itertuples(index=False):
        ticker = row.ticker
        if ticker not in bars:
            ret = np.nan
            buy_close = np.nan
            sell_open = np.nan
            status = "missing_bars"
        else:
            b = ob._clean_ohlc_frame(bars[ticker])
            hist_end = pd.to_datetime(row.history_end).tz_localize(None).normalize()
            if hist_end not in b.index:
                ret = np.nan
                buy_close = np.nan
                sell_open = np.nan
                status = "missing_close"
            else:
                pos = b.index.get_loc(hist_end)
                if isinstance(pos, slice) or pos + 1 >= len(b):
                    ret = np.nan
                    buy_close = float(b.loc[hist_end, "Close"])
                    sell_open = np.nan
                    status = "pending_next_open"
                else:
                    next_date = b.index[pos + 1]
                    buy_close = float(b.loc[hist_end, "Close"])
                    sell_open = float(b.iloc[pos + 1]["Open"])
                    ret = sell_open / buy_close - 1.0
                    status = "settled"
        weighted = float(row.weight) * ret if np.isfinite(ret) else np.nan
        gross_parts.append(weighted)
        rows.append(dict(
            plan_id=row.plan_id,
            plan_date=row.plan_date,
            ticker=ticker,
            weight=float(row.weight),
            buy_close=buy_close,
            sell_open=sell_open,
            overnight_ret=ret,
            weighted_ret=weighted,
            status=status,
        ))
    detail = pd.DataFrame(rows)
    settled = detail["status"].eq("settled").all()
    weights = plan.set_index("ticker")["weight"]
    cbps = ob.realized_cost_bps(weights, cost_bps=cost_bps, cost_by_ticker=cost_by_ticker)
    gross = float(np.nansum(gross_parts)) if settled else np.nan
    net = gross - cbps / 1e4 if settled else np.nan
    summary = pd.DataFrame([dict(
        plan_id=plan["plan_id"].iloc[0],
        plan_date=plan["plan_date"].iloc[0],
        n_assets=int(len(plan)),
        gross=gross,
        net=net,
        gross_bps=gross * 1e4 if np.isfinite(gross) else np.nan,
        net_bps=net * 1e4 if np.isfinite(net) else np.nan,
        cost_bps=float(cbps),
        status="settled" if settled else "pending",
    )])
    return detail, summary


def rolling_report(settlements, windows=(5, 20, 60)):
    """Compute rolling shadow metrics from settlement summaries."""
    if settlements.empty:
        return pd.DataFrame()
    s = settlements[settlements["status"].eq("settled")].copy()
    if s.empty:
        return pd.DataFrame()
    s["plan_date"] = pd.to_datetime(s["plan_date"])
    s = s.sort_values("plan_date")
    rows = []
    for w in windows:
        tail = s.tail(int(w))
        if tail.empty:
            continue
        r = tail["net"].astype(float)
        eq = (1.0 + r).cumprod()
        dd = ((eq.cummax() - eq) / eq.cummax()).max() if len(eq) else np.nan
        rows.append(dict(
            window=int(w),
            n=int(len(tail)),
            mean_bps=float(r.mean() * 1e4),
            win_rate_pct=float((r > 0).mean() * 100),
            total_pct=float((eq.iloc[-1] - 1.0) * 100),
            max_drawdown_pct=float(dd * 100),
            last_plan_date=s["plan_date"].iloc[-1].date().isoformat(),
        ))
    return pd.DataFrame(rows)


def print_report(report):
    """Print rolling diary report."""
    print("== overnight shadow diary report ==")
    if report.empty:
        print("No settled plans yet.")
    else:
        print(report.round(3).to_string(index=False))
    print("Caveat: shadow diary uses observed close/open prices, not actual auction fills.")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Shadow diary for overnight ETF basket plans.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--tickers", default="expanded")
        p.add_argument("--start", default="2019-01-01")
        p.add_argument("--end")
        p.add_argument("--provider", choices=["auto", "polygon", "yfinance"], default="auto")
        p.add_argument("--out", default="paper_logs")

    p_plan = sub.add_parser("plan")
    add_common(p_plan)
    p_plan.add_argument("--rule", default="mean")
    p_plan.add_argument("--lookback", type=int, default=40)
    p_plan.add_argument("--top-n", type=int, default=5)
    p_plan.add_argument("--weighting", default="equal")
    p_plan.add_argument("--cap", type=float, default=0.35)
    p_plan.add_argument("--notional", type=float, default=10_000.0)
    p_plan.add_argument("--signal-kind", choices=["overnight", "close"], default="overnight")
    p_plan.add_argument("--plan-date")

    p_settle = sub.add_parser("settle")
    add_common(p_settle)
    p_settle.add_argument("--plan-id")
    p_settle.add_argument("--cost-bps", type=float, default=1.0)
    p_settle.add_argument("--cost-model", choices=["flat", "etf"], default="etf")

    p_report = sub.add_parser("report")
    p_report.add_argument("--out", default="paper_logs")

    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.cmd == "plan":
        tickers = ob.parse_tickers(args.tickers)
        bars, provider = _latest_bars(tickers, args.start, args.end, args.provider)
        plan = create_plan(bars, rule=args.rule, lookback=args.lookback, top_n=args.top_n,
                           universe=tickers, weighting=args.weighting, cap=args.cap,
                           notional=args.notional, signal_kind=args.signal_kind,
                           plan_date=args.plan_date)
        if plan.empty:
            print("No plan produced.")
            return
        all_plans = _append_csv(plan, out / PLAN_FILE, key_cols=["plan_id", "ticker"])
        print(f"== plan [{provider}] ==")
        print(plan.round(6).to_string(index=False))
        print(f"wrote {len(plan)} rows to {out / PLAN_FILE} ({len(all_plans)} total rows)")
        return

    if args.cmd == "settle":
        plans = _read_csv(out / PLAN_FILE)
        if plans.empty:
            raise SystemExit("No plans found.")
        plan_id = args.plan_id or plans["plan_id"].iloc[-1]
        plan = plans[plans["plan_id"].eq(plan_id)].copy()
        tickers = sorted(plan["ticker"].unique())
        bars, provider = _latest_bars(tickers, args.start, args.end, args.provider)
        detail, summary = settle_plan(
            plan, bars, cost_bps=args.cost_bps,
            cost_by_ticker=ob.DEFAULT_ETF_COST_BPS if args.cost_model == "etf" else None,
        )
        _append_csv(detail, out / "overnight_settlement_details.csv", key_cols=["plan_id", "ticker"])
        settlements = _append_csv(summary, out / SETTLE_FILE, key_cols=["plan_id"])
        report = rolling_report(settlements)
        report.to_csv(out / REPORT_FILE, index=False)
        print(f"== settle [{provider}] ==")
        print(summary.round(6).to_string(index=False))
        print_report(report)
        return

    if args.cmd == "report":
        settlements = _read_csv(out / SETTLE_FILE)
        report = rolling_report(settlements)
        report.to_csv(out / REPORT_FILE, index=False)
        print_report(report)


if __name__ == "__main__":
    main()
