"""把研究信号/回测 pick 转成 paper orders 的小工具。"""
from __future__ import annotations

from paper_broker import LocalPaperBroker, OrderIntent, RiskGate


def intents_from_picks(res, notional_per_trade=1_000.0):
    """把 backtest/signal_lab 的逐槽 pick 转成 long-only buy 意图。

    注意:这只是 forward-test 的订单桥,不是可交易性证明;真正下单前仍要过 falsify + 风控。
    """
    intents = []
    for row in res.itertuples(index=False):
        reason = f"date={getattr(row, 'date', '')} slot={getattr(row, 'slot', '')}"
        intents.append(OrderIntent(symbol=row.pick, side="buy", notional=float(notional_per_trade),
                                   reason=reason))
    return intents


def run_intents(broker: LocalPaperBroker, intents, prices=None, risk_gate: RiskGate | None = None):
    """按给定 prices 依次执行订单,返回成交/拒单列表。"""
    prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
    out = []
    for intent in intents:
        out.append(broker.submit_order(intent, price=prices.get(intent.symbol), risk_gate=risk_gate))
    return out


def broker_summary(broker: LocalPaperBroker, prices=None):
    """返回纸面账户摘要,方便测试和 notebook 使用。"""
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
    """打印纸面账户快照。"""
    s = broker_summary(broker, prices)
    print("== Paper account ==")
    print(f"cash {s['cash']:.2f} | equity {s['equity']:.2f} | gross {s['gross_exposure']:.2f}")
    print(f"positions {s['n_positions']} | fills {s['n_fills']}")
    pos = broker.positions_frame(prices)
    if not pos.empty:
        print("\n持仓:")
        print(pos.round(4).to_string(index=False))
    fills = broker.fills_frame()
    if not fills.empty:
        print("\n最近成交:")
        print(fills.tail(10).to_string(index=False))
