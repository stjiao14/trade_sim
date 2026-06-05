import pytest

from paper_broker import (
    AlpacaPaperBroker,
    LocalPaperBroker,
    OrderIntent,
    RiskGate,
    RiskLimits,
)
from paper_runner import broker_summary, run_intents


def test_local_paper_buy_sell_and_equity():
    broker = LocalPaperBroker(cash=10_000, price_map={"AAPL": 100.0}, slippage_bps=10.0)
    buy = broker.submit_order(OrderIntent("aapl", "buy", notional=1_000.0))
    assert buy.status == "filled"
    assert abs(buy.price - 100.10) < 1e-9
    assert broker.position_qty("AAPL") > 9.99

    sell = broker.submit_order(OrderIntent("AAPL", "sell", qty=5.0), price=110.0)
    assert sell.status == "filled"
    assert broker.position_qty("AAPL") > 4.99
    assert broker.equity({"AAPL": 110.0}) > 10_000


def test_risk_gate_blocks_large_order_and_short():
    broker = LocalPaperBroker(cash=5_000, price_map={"MSFT": 200.0})
    gate = RiskGate(RiskLimits(max_order_notional=1_000.0, allow_short=False))

    big = broker.submit_order(OrderIntent("MSFT", "buy", notional=2_000.0), risk_gate=gate)
    assert big.status == "rejected"
    assert broker.position_qty("MSFT") == 0

    short = broker.submit_order(OrderIntent("MSFT", "sell", qty=1.0), risk_gate=gate)
    assert short.status == "rejected"
    assert "short" in short.message


def test_runner_executes_intents_and_summary():
    broker = LocalPaperBroker(cash=20_000, price_map={"AAPL": 100.0, "MSFT": 200.0})
    gate = RiskGate(RiskLimits(max_order_notional=5_000.0, max_gross_notional=10_000.0))
    fills = run_intents(
        broker,
        [OrderIntent("AAPL", "buy", notional=1_000), OrderIntent("MSFT", "buy", qty=3)],
        risk_gate=gate,
    )
    assert [f.status for f in fills] == ["filled", "filled"]
    s = broker_summary(broker)
    assert s["n_positions"] == 2
    assert s["n_fills"] == 2
    assert abs(s["gross_exposure"] - 1_600.0) < 1e-9


def test_alpaca_adapter_is_paper_only_and_requires_keys(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValueError, match="paper endpoint"):
        AlpacaPaperBroker(api_key="k", secret_key="s", base_url="https://api.alpaca.markets")
    with pytest.raises(ValueError, match="missing"):
        AlpacaPaperBroker()
