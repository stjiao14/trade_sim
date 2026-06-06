"""Broker API 延迟/稳定性小基准。

默认只测 Alpaca paper 的 read endpoints;只有显式 --submit-test-order 才会发测试单并尝试取消。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paper_broker import AlpacaPaperBroker, OrderIntent


def time_call(name, fn, n=5):
    """重复调用 fn,记录每次 latency/error。"""
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
    """按 endpoint 汇总 p50/p95/max/error_rate。"""
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
    """对 {name: fn} 运行 benchmark,返回 (raw, summary)。"""
    rows = []
    for name, fn in callables.items():
        rows.extend(time_call(name, fn, n=n))
    raw = pd.DataFrame(rows)
    return raw, summarize(rows)


def alpaca_callables(broker, submit_test_order=False, symbol="AAPL", notional=1.0):
    """构造 Alpaca paper benchmark endpoints。submit_test_order 默认关闭。"""
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
    """运行 Alpaca paper benchmark。"""
    broker = AlpacaPaperBroker()
    return benchmark_callables(alpaca_callables(broker, submit_test_order, symbol, notional), n=n)


def write_reports(raw, summary, out_dir="broker_benchmark_logs"):
    """写出 raw + summary CSV。"""
    p = Path(out_dir); p.mkdir(parents=True, exist_ok=True)
    raw.to_csv(p / "broker_latency_raw.csv", index=False)
    summary.to_csv(p / "broker_latency_summary.csv", index=False)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description="Broker API latency benchmark。默认不下单。")
    ap.add_argument("--broker", choices=["alpaca-paper"], default="alpaca-paper")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="broker_benchmark_logs")
    ap.add_argument("--submit-test-order", action="store_true",
                    help="显式打开后才会发 Alpaca paper 测试单并尝试取消。")
    ap.add_argument("--symbol", default="AAPL")
    ap.add_argument("--notional", type=float, default=1.0)
    args = ap.parse_args(argv)

    raw, summary = run_alpaca(n=args.n, submit_test_order=args.submit_test_order,
                              symbol=args.symbol, notional=args.notional)
    out = write_reports(raw, summary, args.out)
    print(summary.round(2).to_string(index=False))
    print(f"\nlogs={out}")
    if args.submit_test_order:
        print("注意:本次包含 Alpaca paper 测试单提交/取消。")
    print("免责声明:API benchmark 只衡量当前网络/账户/API 状态,不构成投资建议。")


if __name__ == "__main__":
    main()
