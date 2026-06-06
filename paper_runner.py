"""Small helpers that convert research/backtest picks into paper orders."""
from __future__ import annotations

from paper_broker import LocalPaperBroker, OrderIntent, RiskGate


def intents_from_picks(res, notional_per_trade=1_000.0):
    """Convert slot-level picks from backtest/signal_lab into long-only buy intents.

    This is only a forward-test order bridge, not proof of tradability. Real
    execution still needs falsification and risk checks.
    """
    intents = []
    for row in res.itertuples(index=False):
        reason = f"date={getattr(row, 'date', '')} slot={getattr(row, 'slot', '')}"
        intents.append(OrderIntent(symbol=row.pick, side="buy", notional=float(notional_per_trade),
                                   reason=reason))
    return intents


def run_intents(broker: LocalPaperBroker, intents, prices=None, risk_gate: RiskGate | None = None):
    """Execute orders sequentially with optional prices and return fills/rejections."""
    prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
    out = []
    for intent in intents:
        out.append(broker.submit_order(intent, price=prices.get(intent.symbol), risk_gate=risk_gate))
    return out


def broker_summary(broker: LocalPaperBroker, prices=None):
    """Return a paper account summary for tests and notebooks."""
    pos = broker.positions_frame(prices)
    gross = float(pos["market_value"].abs().sum()) if not pos.empty else 0.0
    return dict(
        cash=float(broker.cash),
        equity=float(broker.equity(prices)),
        gross_exposure=float(gross),
        n_positions=int(len(pos)),
        n_fills=int(len(broker.fills)),
    )


def print_broker_report(broker: LocalPaperBroker, prices=None):
    """Print a paper account snapshot."""
    s = broker_summary(broker, prices)
    print("== Paper account ==")
    print(f"cash {s['cash']:.2f} | equity {s['equity']:.2f} | gross {s['gross_exposure']:.2f}")
    print(f"positions {s['n_positions']} | fills {s['n_fills']}")
    pos = broker.positions_frame(prices)
    if not pos.empty:
        print("\nPositions:")
        print(pos.round(4).to_string(index=False))
    fills = broker.fills_frame()
    if not fills.empty:
        print("\nRecent fills:")
        print(fills.tail(10).to_string(index=False))
