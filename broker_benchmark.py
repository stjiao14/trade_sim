"""Small broker API latency/reliability benchmark.

By default it only reads Alpaca paper endpoints. It submits and cancels a test
order only when --submit-test-order is explicitly set.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_broker import AlpacaPaperBroker, OrderIntent


def time_call(name, fn, n=5):
    """Call fn repeatedly and record per-call latency/errors."""
    rows = []
    for i in range(n):
        t0 = time.perf_counter()
        ok = True
        err = ""
        try:
            fn()
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - t0) * 1000
        rows.append(dict(endpoint=name, run=i, ok=ok, latency_ms=float(latency_ms), error=err))
    return rows


def summarize(rows):
    """Summarize p50/p95/max/error_rate by endpoint."""
    df = pd.DataFrame(rows)
    out = []
    for endpoint, g in df.groupby("endpoint"):
        lat = g.loc[g["ok"], "latency_ms"].astype(float)
        out.append(dict(
            endpoint=endpoint,
            n=int(len(g)),
            ok=int(g["ok"].sum()),
            error_rate_pct=float((~g["ok"]).mean() * 100),
            p50_ms=float(np.percentile(lat, 50)) if len(lat) else float("nan"),
            p95_ms=float(np.percentile(lat, 95)) if len(lat) else float("nan"),
            max_ms=float(lat.max()) if len(lat) else float("nan"),
        ))
    return pd.DataFrame(out).sort_values("endpoint").reset_index(drop=True)


def benchmark_callables(callables, n=5):
    """Benchmark {name: fn} callables and return (raw, summary)."""
    rows = []
    for name, fn in callables.items():
        rows.extend(time_call(name, fn, n=n))
    raw = pd.DataFrame(rows)
    return raw, summarize(rows)


def alpaca_callables(broker, submit_test_order=False, symbol="AAPL", notional=1.0):
    """Build Alpaca paper benchmark endpoints. submit_test_order is off by default."""
    calls = {
        "account": broker.get_account,
        "positions": broker.get_positions,
        "orders": broker.get_orders,
        "fills": broker.get_fills,
    }
    if submit_test_order:
        def submit_and_cancel():
            resp = broker.submit_order(OrderIntent(symbol, "buy", notional=notional))
            oid = resp.get("id")
            if oid:
                broker.cancel_order(oid)
            return resp
        calls["submit_cancel_order"] = submit_and_cancel
    return calls


def run_alpaca(n=5, submit_test_order=False, symbol="AAPL", notional=1.0):
    """Run Alpaca paper benchmark."""
    broker = AlpacaPaperBroker()
    return benchmark_callables(alpaca_callables(broker, submit_test_order, symbol, notional), n=n)


def write_reports(raw, summary, out_dir="broker_benchmark_logs"):
    """Write raw and summary CSV reports."""
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    raw.to_csv(p / "broker_latency_raw.csv", index=False)
    summary.to_csv(p / "broker_latency_summary.csv", index=False)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="Broker API latency benchmark. No orders by default.")
    ap.add_argument("--broker", choices=["alpaca-paper"], default="alpaca-paper")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="broker_benchmark_logs")
    ap.add_argument("--submit-test-order", action="store_true",
                    help="Only when set: submit an Alpaca paper test order and try to cancel it.")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--notional", type=float, default=1.0)
    args = ap.parse_args(argv)

    raw, summary = run_alpaca(n=args.n, submit_test_order=args.submit_test_order,
                              symbol=args.symbol, notional=args.notional)
    out = write_reports(raw, summary, args.out)
    print(summary.round(2).to_string(index=False))
    print(f"\nlogs={out}")
    if args.submit_test_order:
        print("Note: this run submitted/cancelled an Alpaca paper test order.")
    print("Disclaimer: this benchmark only measures current network/account/API state; it is not investment advice.")


if __name__ == "__main__":
    main()
