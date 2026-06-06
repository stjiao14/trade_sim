"""Paper-trading execution layer: local fills plus Alpaca paper adapter.

This module does not judge signal quality. It turns already-researched order
intents into auditable paper fills.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Mapping

import pandas as pd


PAPER_ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


def _local_config_value(name, default=None):
    """Read a local fallback value from git-ignored config_local.py."""
    try:
        import config_local as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default


@dataclass(frozen=True)
class OrderIntent:
    """Order intent from a strategy. Exactly one of qty/notional is required."""

    symbol: str
    side: str
    qty: float | None = None
    notional: float | None = None
    order_type: str = "market"
    limit_price: float | None = None
    reason: str = ""

    def __post_init__(self):
        symbol = self.symbol.upper().strip()
        side = self.side.lower().strip()
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        has_qty = self.qty is not None and self.qty > 0
        has_notional = self.notional is not None and self.notional > 0
        if has_qty == has_notional:
            raise ValueError("exactly one of positive qty or notional is required")
        if self.order_type not in {"market", "limit"}:
            raise ValueError("order_type must be market or limit")
        if self.order_type == "limit" and (self.limit_price is None or self.limit_price <= 0):
            raise ValueError("limit orders require a positive limit_price")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)


@dataclass(frozen=True)
class RiskLimits:
    """Minimal paper risk limits."""

    max_order_notional: float = 10_000.0
    max_symbol_notional: float = 25_000.0
    max_gross_notional: float = 100_000.0
    min_cash: float = 0.0
    allow_short: bool = False
    blocked_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fill:
    """Order fill or rejection result. Rejections have zero qty/notional."""

    timestamp: str
    symbol: str
    side: str
    qty: float
    price: float
    notional: float
    commission: float
    status: str
    reason: str = ""
    message: str = ""


class RiskGate:
    """Hard risk checks before an order reaches the broker."""

    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check_order(self, broker, intent: OrderIntent, qty: float, price: float):
        """Return (ok, message), using fill-price estimates to prevent risk drift."""
        symbol = intent.symbol
        notional = abs(qty * price)
        if symbol in {s.upper() for s in self.limits.blocked_symbols}:
            return False, f"{symbol} is blocked"
        if notional > self.limits.max_order_notional:
            return False, "order notional exceeds max_order_notional"

        signed = qty if intent.side == "buy" else -qty
        projected_qty = broker.position_qty(symbol) + signed
        if not self.limits.allow_short and projected_qty < -1e-9:
            return False, "short sale not allowed"

        projected_symbol_notional = abs(projected_qty * price)
        if projected_symbol_notional > self.limits.max_symbol_notional:
            return False, "symbol exposure exceeds max_symbol_notional"

        gross_now = broker.gross_exposure()
        current_symbol = abs(broker.position_qty(symbol) * price)
        projected_gross = gross_now - current_symbol + projected_symbol_notional
        if projected_gross > self.limits.max_gross_notional:
            return False, "gross exposure exceeds max_gross_notional"

        if intent.side == "buy" and broker.cash - qty * price < self.limits.min_cash:
            return False, "cash would fall below min_cash"
        return True, "ok"


class LocalPaperBroker:
    """Offline paper broker using provided prices, with cash/positions/fills/orders."""

    def __init__(self, cash=100_000.0, price_map=None, slippage_bps=0.0, commission_bps=0.0):
        self.cash = float(cash)
        self.price_map = {k.upper(): float(v) for k, v in (price_map or {}).items()}
        self.slippage_bps = float(slippage_bps)
        self.commission_bps = float(commission_bps)
        self.positions: dict[str, float] = {}
        self.orders: list[dict] = []
        self.fills: list[Fill] = []

    def position_qty(self, symbol):
        return float(self.positions.get(symbol.upper(), 0.0))

    def get_price(self, symbol, override=None):
        if override is not None:
            return float(override)
        symbol = symbol.upper()
        if symbol not in self.price_map:
            raise KeyError(f"missing paper price for {symbol}")
        return float(self.price_map[symbol])

    def gross_exposure(self, price_map: Mapping[str, float] | None = None):
        px = {**self.price_map, **{k.upper(): float(v) for k, v in (price_map or {}).items()}}
        return float(sum(abs(q) * px.get(sym, 0.0) for sym, q in self.positions.items()))

    def equity(self, price_map: Mapping[str, float] | None = None):
        px = {**self.price_map, **{k.upper(): float(v) for k, v in (price_map or {}).items()}}
        return float(self.cash + sum(q * px.get(sym, 0.0) for sym, q in self.positions.items()))

    def submit_order(self, intent: OrderIntent, price=None, timestamp=None, risk_gate: RiskGate | None = None):
        """Fill at the current paper price. Risk rejections do not mutate account state."""
        ts = timestamp or datetime.now(timezone.utc).isoformat()
        raw_price = self.get_price(intent.symbol, price)
        qty = float(intent.qty if intent.qty is not None else intent.notional / raw_price)
        slip = self.slippage_bps / 1e4
        fill_price = raw_price * (1 + slip if intent.side == "buy" else 1 - slip)

        self.orders.append({**asdict(intent), "timestamp": ts, "reference_price": raw_price})
        if risk_gate is not None:
            ok, msg = risk_gate.check_order(self, intent, qty, fill_price)
            if not ok:
                return Fill(ts, intent.symbol, intent.side, 0.0, fill_price, 0.0, 0.0,
                            "rejected", intent.reason, msg)

        notional = qty * fill_price
        commission = abs(notional) * self.commission_bps / 1e4
        signed_qty = qty if intent.side == "buy" else -qty
        cash_delta = -notional - commission if intent.side == "buy" else notional - commission
        self.positions[intent.symbol] = self.position_qty(intent.symbol) + signed_qty
        if abs(self.positions[intent.symbol]) < 1e-12:
            del self.positions[intent.symbol]
        self.cash += cash_delta

        fill = Fill(ts, intent.symbol, intent.side, qty, fill_price, notional, commission,
                    "filled", intent.reason, "")
        self.fills.append(fill)
        return fill

    def fills_frame(self):
        return pd.DataFrame([asdict(f) for f in self.fills])

    def orders_frame(self):
        return pd.DataFrame(self.orders)

    def positions_frame(self, price_map: Mapping[str, float] | None = None):
        px = {**self.price_map, **{k.upper(): float(v) for k, v in (price_map or {}).items()}}
        rows = []
        for sym, qty in sorted(self.positions.items()):
            price = px.get(sym, float("nan"))
            rows.append(dict(symbol=sym, qty=float(qty), price=float(price), market_value=float(qty * price)))
        return pd.DataFrame(rows)


class AlpacaPaperBroker:
    """Alpaca paper trading adapter. Only the paper endpoint is allowed."""

    def __init__(self, api_key=None, secret_key=None, base_url=PAPER_ALPACA_BASE_URL):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY") or _local_config_value("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY") or _local_config_value("ALPACA_SECRET_KEY")
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v2"):
            self.base_url = self.base_url[:-3]
        if self.base_url != PAPER_ALPACA_BASE_URL:
            raise ValueError("AlpacaPaperBroker only supports the paper endpoint")
        if not self.api_key or not self.secret_key:
            raise ValueError("missing ALPACA_API_KEY / ALPACA_SECRET_KEY")

    def _request(self, method, path, body=None, params=None):
        """Alpaca paper REST request. path should be /v2/...."""
        url = f"{self.base_url}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca paper request failed: {exc.code} {detail}") from exc

    def get_account(self):
        """Read paper account cash/equity/buying_power and related fields."""
        return self._request("GET", "/v2/account")

    def get_positions(self):
        """Read current Alpaca paper positions."""
        return self._request("GET", "/v2/positions")

    def get_orders(self, status="all", limit=100, direction="desc"):
        """Read paper orders. Defaults to all for CSV reconciliation."""
        return self._request("GET", "/v2/orders",
                             params=dict(status=status, limit=limit, direction=direction))

    def get_activities(self, activity_type="FILL", limit=100, direction="desc"):
        """Read account activities. activity_type='FILL' approximates fills."""
        return self._request("GET", f"/v2/account/activities/{activity_type}",
                             params=dict(limit=limit, direction=direction))

    def get_fills(self, limit=100, direction="desc"):
        """Read fill activities."""
        return self.get_activities("FILL", limit=limit, direction=direction)

    def cancel_order(self, order_id):
        """Cancel one paper order."""
        return self._request("DELETE", f"/v2/orders/{order_id}")

    def submit_order(self, intent: OrderIntent, time_in_force="day"):
        """Submit to Alpaca paper API. Network calls happen only when this adapter is used."""
        body = {
            "symbol": intent.symbol,
            "side": intent.side,
            "type": intent.order_type,
            "time_in_force": time_in_force,
        }
        if intent.qty is not None:
            body["qty"] = str(intent.qty)
        else:
            body["notional"] = str(intent.notional)
        if intent.limit_price is not None:
            body["limit_price"] = str(intent.limit_price)
        return self._request("POST", "/v2/orders", body=body)

    def snapshot(self):
        """Read account/positions/orders/fills for paper reconciliation."""
        return dict(
            account=self.get_account(),
            positions=self.get_positions(),
            orders=self.get_orders(),
            fills=self.get_fills(),
        )
