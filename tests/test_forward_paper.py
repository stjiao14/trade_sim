import pandas as pd

import forward_paper as fp
import intraday_seasonality_backtest as bt
from paper_broker import LocalPaperBroker, RiskGate, RiskLimits


def _bars(n_days=45):
    tickers = ["AAA", "BBB", "CCC"]
    days = pd.bdate_range("2026-01-02", periods=n_days, tz=bt.TZ)
    out = {}
    for ti, t in enumerate(tickers):
        ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30 * k)
                               for d in days for k in range(13)])
        slot = list(range(13)) * len(days)
        ret = [0.0004 * (ti == (s % len(tickers))) for s in slot]
        out[t] = pd.DataFrame({"Open": 100.0, "Close": 100.0 * (1 + pd.Series(ret).values)}, index=ts)
    return out


def test_next_slot_picks_generates_one_order_per_slot():
    lr = bt.keep_full_sessions(bt.to_slot_returns(_bars()))
    plan = fp.next_slot_picks(lr, lookback=30)
    assert len(plan) == 13
    assert set(plan.columns) >= {"slot", "pick", "score_bps", "lookback_start", "lookback_end"}


def test_forward_shadow_does_not_change_broker():
    bars = _bars()
    lr = bt.keep_full_sessions(bt.to_slot_returns(bars))
    prices = fp.latest_close_prices(bars)
    broker = LocalPaperBroker(cash=10_000, price_map=prices)
    run = fp.run_forward_local(lr, prices, mode="shadow", broker=broker)
    assert run["executed"] is False
    assert run["summary"]["n_fills"] == 0
    assert len(run["orders"]) == 13


def test_forward_paper_executes_with_risk_gate():
    bars = _bars()
    lr = bt.keep_full_sessions(bt.to_slot_returns(bars))
    prices = fp.latest_close_prices(bars)
    broker = LocalPaperBroker(cash=20_000, price_map=prices)
    gate = RiskGate(RiskLimits(max_order_notional=2_000, max_gross_notional=20_000))
    run = fp.run_forward_local(lr, prices, mode="paper", notional_per_slot=500,
                               broker=broker, risk_gate=gate)
    assert run["executed"] is True
    assert run["summary"]["n_fills"] == 13
    assert run["summary"]["gross_exposure"] > 0


def test_write_forward_logs(tmp_path):
    bars = _bars()
    lr = bt.keep_full_sessions(bt.to_slot_returns(bars))
    prices = fp.latest_close_prices(bars)
    run = fp.run_forward_local(lr, prices, mode="shadow")
    out = fp.write_forward_logs(run, tmp_path)
    assert (out / "paper_orders.csv").exists()
    assert (out / "paper_research.csv").exists()
