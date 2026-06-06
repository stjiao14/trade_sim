"""Forward paper runner: load data, build next-day orders, then shadow/paper execute.

Default mode is shadow and does not touch live trading. This is an execution
sandbox after research, not proof of alpha.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import intraday_seasonality_backtest as bt
from paper_broker import AlpacaPaperBroker, LocalPaperBroker, OrderIntent, RiskGate, RiskLimits
from paper_runner import broker_summary, print_broker_report


def latest_close_prices(bars):
    """Read the latest close for each ticker from {ticker: bars} for local fills."""
    out = {}
    for t, df in bars.items():
        if df.empty:
            continue
        out[t.upper()] = float(df["Close"].dropna().iloc[-1])
    return out


def load_polygon_panel(tickers=bt.UNIVERSE, start=None, end=None, api_key=None):
    """Load Polygon 30m bars and convert them into a full-session panel."""
    if end is None:
        end = pd.Timestamp.now(tz=bt.TZ).date().isoformat()
    if start is None:
        start = (pd.Timestamp(end) - pd.DateOffset(months=18)).date().isoformat()
    bars = bt.load_bars_polygon(tickers, start, end, api_key=api_key)
    lr = bt.keep_full_sessions(bt.to_slot_returns(bars))
    return bars, lr


def next_slot_picks(lr, lookback=bt.LOOKBACK):
    """Use the latest lookback full sessions to build the next day's 13-slot plan."""
    piv = bt._pivot(lr)
    dates = sorted(lr["date"].unique())
    if len(dates) < lookback:
        raise ValueError("not enough full sessions for forward plan")
    hist_dates = dates[-lookback:]
    dlev = piv.index.get_level_values("date")
    sub = piv[dlev.isin(set(hist_dates))]
    rows = []
    for slot in sorted(lr["slot"].unique()):
        tr = sub.xs(slot, level="slot")
        sc = tr.mean().dropna()
        if sc.empty:
            continue
        pick = sc.idxmax()
        rows.append(dict(
            slot=int(slot),
            pick=str(pick),
            score_bps=float(sc.loc[pick] * 1e4),
            lookback_start=hist_dates[0],
            lookback_end=hist_dates[-1],
            n_lookback_days=int(len(hist_dates)),
        ))
    return pd.DataFrame(rows)


def research_verdict(lr, strict=True, random_seeds=5):
    """Lightweight research gate. strict=True is closer to signal_lab's hard verdict."""
    m = bt.evaluate(lr, random_seeds=random_seeds)
    gates = dict(
        raw_net_positive=m["raw_net_bps"] > 0,
        seasonality_above_floor=m["season_excess_bps"] > 3.0,
        concentration_ok=m["concentration_pct"] < 30.0,
        daily_ci_positive=m["daily_boot_lo_bps"] > 0,
    )
    if strict:
        gates["regime_not_one_sided"] = not (m["regime_hi_bps"] > 0 and m["regime_lo_bps"] <= 0)
    reasons = [k for k, ok in gates.items() if not ok]
    return dict(verdict="PASS" if not reasons else "FAIL", fail_reasons=reasons, metrics=m, gates=gates)


def intents_from_plan(plan, notional_per_slot=1_000.0):
    """Convert next_slot_picks output into order intents."""
    out = []
    for row in plan.itertuples(index=False):
        out.append(OrderIntent(
            symbol=row.pick,
            side="buy",
            notional=float(notional_per_slot),
            reason=f"forward slot={row.slot} score={row.score_bps:+.2f}bps",
        ))
    return out


def default_risk_gate(config=None):
    """Build default risk checks from a dict, usually config_local.PAPER_TRADING."""
    c = config or {}
    limits = RiskLimits(
        max_order_notional=float(c.get("max_order_notional", 5_000.0)),
        max_symbol_notional=float(c.get("max_symbol_notional", 20_000.0)),
        max_gross_notional=float(c.get("max_gross_notional", 100_000.0)),
        min_cash=float(c.get("min_cash", 0.0)),
        allow_short=bool(c.get("allow_short", False)),
        blocked_symbols=tuple(c.get("blocked_symbols", [])),
    )
    return RiskGate(limits)


def run_forward_local(lr, prices, mode="shadow", notional_per_slot=1_000.0,
                      broker=None, risk_gate=None, require_pass=False):
    """Build the next-day plan and optionally execute against local paper broker.

    mode='shadow' records intents only; mode='paper' mutates the local broker.
    require_pass=True skips execution when the research gate fails.
    """
    plan = next_slot_picks(lr)
    verdict = research_verdict(lr)
    intents = intents_from_plan(plan, notional_per_slot=notional_per_slot)
    execute = mode == "paper" and (not require_pass or verdict["verdict"] == "PASS")
    broker = broker or LocalPaperBroker(price_map=prices)
    risk_gate = risk_gate or default_risk_gate()
    fills = []
    if execute:
        for intent in intents:
            fills.append(broker.submit_order(intent, price=prices.get(intent.symbol), risk_gate=risk_gate))
    orders = pd.DataFrame([{**asdict(i), "mode": mode, "research_verdict": verdict["verdict"]}
                           for i in intents])
    return dict(plan=plan, orders=orders, fills=fills, broker=broker,
                verdict=verdict, executed=bool(execute), summary=broker_summary(broker, prices))


def run_forward_alpaca(lr, mode="shadow", notional_per_slot=1_000.0,
                       broker=None, require_pass=False):
    """Build the next-day plan and optionally submit to Alpaca paper. Shadow sends nothing."""
    plan = next_slot_picks(lr)
    verdict = research_verdict(lr)
    intents = intents_from_plan(plan, notional_per_slot=notional_per_slot)
    execute = mode == "paper" and (not require_pass or verdict["verdict"] == "PASS")
    broker = broker or AlpacaPaperBroker()
    responses = []
    if execute:
        for intent in intents:
            responses.append(broker.submit_order(intent))
    orders = pd.DataFrame([{**asdict(i), "mode": mode, "broker": "alpaca-paper",
                            "research_verdict": verdict["verdict"]} for i in intents])
    summary = dict(n_orders=int(len(orders)), n_responses=int(len(responses)))
    return dict(plan=plan, orders=orders, fills=responses, broker=broker,
                verdict=verdict, executed=bool(execute), summary=summary)


def write_forward_logs(run, out_dir="paper_logs"):
    """Write a forward run to CSV: orders, fills, plan, summary, and research."""
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    run["plan"].to_csv(p / "paper_plan.csv", index=False)
    run["orders"].to_csv(p / "paper_orders.csv", index=False)
    fills = pd.DataFrame([asdict(f) if hasattr(f, "__dataclass_fields__") else f for f in run["fills"]])
    fills.to_csv(p / "paper_fills.csv", index=False)
    pd.DataFrame([run["summary"]]).to_csv(p / "paper_summary.csv", index=False)
    m = run["verdict"]["metrics"]
    flat = {k: v for k, v in m.items() if np.isscalar(v)}
    flat["research_verdict"] = run["verdict"]["verdict"]
    flat["fail_reasons"] = ",".join(run["verdict"]["fail_reasons"])
    pd.DataFrame([flat]).to_csv(p / "paper_research.csv", index=False)
    return p


def _paper_config():
    try:
        import config_local as cfg
    except Exception:
        import config_example as cfg
    return getattr(cfg, "PAPER_TRADING", {})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Forward paper runner. Shadow mode does not fill orders.")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--mode", choices=["shadow", "paper"], default="shadow")
    ap.add_argument("--broker", choices=["local", "alpaca-paper"], default="local")
    ap.add_argument("--notional", type=float, default=1_000.0)
    ap.add_argument("--require-pass", action="store_true")
    ap.add_argument("--out", default="paper_logs")
    args = ap.parse_args(argv)

    cfg = _paper_config()
    bars, lr = load_polygon_panel(start=args.start, end=args.end)
    prices = latest_close_prices(bars)
    if args.broker == "alpaca-paper":
        run = run_forward_alpaca(lr, mode=args.mode, notional_per_slot=args.notional,
                                 broker=AlpacaPaperBroker(base_url=cfg.get("alpaca_base_url",
                                                                            "https://paper-api.alpaca.markets")),
                                 require_pass=args.require_pass)
    else:
        broker = LocalPaperBroker(
            cash=float(cfg.get("starting_cash", 100_000.0)),
            price_map=prices,
            slippage_bps=float(cfg.get("slippage_bps", 1.0)),
            commission_bps=float(cfg.get("commission_bps", 0.0)),
        )
        run = run_forward_local(lr, prices, mode=args.mode, notional_per_slot=args.notional,
                                broker=broker, risk_gate=default_risk_gate(cfg),
                                require_pass=args.require_pass)
    out = write_forward_logs(run, args.out)
    v = run["verdict"]
    print(f"research {v['verdict']} | executed={run['executed']} | logs={out}")
    if v["fail_reasons"]:
        print("fail reasons:", ", ".join(v["fail_reasons"]))
    print(run["plan"].to_string(index=False))
    if args.broker == "local":
        print_broker_report(run["broker"], prices)
    else:
        print(f"alpaca responses: {len(run['fills'])}")
    print("\nDisclaimer: paper forward-test is only an execution/logging sandbox, not investment advice or a replacement for signal falsification.")


if __name__ == "__main__":
    main()
