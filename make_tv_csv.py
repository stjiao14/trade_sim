"""Convert profolio.xlsx into TradingView portfolio-import CSV format.

Format (see example.9205072d664d0a6f0bee.csv):
    Symbol,Side,Qty,Fill Price,Commission,Closing Time
    NASDAQ:AAPL,Buy,10,217,0,2024-09-17 0:00:00
    $CASH,Deposit,5000,0,0,2024-08-24 0:00:00

Rules:
- Each non-cash holding becomes one Buy at its average cost where known
  (preserves unrealized P&L), else at current price.
- M1 rows lack quantity: qty = market_value / live_price, priced so the
  position's cost basis equals market_value - unrealized_gain.
- Cash-like rows (SPAXX/FDRXX/QPCTQ/VMFXX) generate no trades; one final
  $CASH Deposit/Withdrawal reconciles TradingView's default 100k cash:
      deposit = actual_cash - 100_000 + total_buy_cost
- Unvested RSU (923 GOOGL) is NOT included: it is not a brokerage asset.

Usage: .venv/bin/python make_tv_csv.py
"""
import pandas as pd
import yfinance as yf

TRADE_DATE = "2026-07-28 0:00:00"
# TV's displayed "portfolio value" excludes its $100k virtual seed cash, so we
# do NOT subtract it from the reconciling deposit: deposit = actual_cash +
# total_buy_cost. Side effect: the cash line shows actual + 100k, while the
# headline portfolio value matches reality. Set back to 100_000 to reverse.
TV_DEFAULT_CASH = 0.0
OUT = "tv_portfolio.csv"

# TradingView exchange prefixes. Verify when importing; TV symbol search
# confirms whether AMEX: vs NYSEARC: is accepted for a given ETF.
EXCHANGE = {
    "QQQM": "NASDAQ", "AMD": "NASDAQ", "AVGO": "NASDAQ", "MSFT": "NASDAQ",
    "INTC": "NASDAQ", "BNTX": "NASDAQ", "ZM": "NASDAQ", "GOOGL": "NASDAQ",
    "TQQQ": "NASDAQ", "QQQ": "NASDAQ",
    "TSM": "NYSE",
    "SPY": "AMEX", "GLD": "AMEX", "DRAM": "AMEX",
    "VOO": "AMEX", "VXF": "AMEX",
}
# 401(k) institutional trusts are not tradable tickers and their unit prices
# do not match the similarly-named mutual funds (VINIX NAV ~$594 vs trust
# ~$115). Map them to proxy ETFs at equal dollar value; cost basis resets to
# current price (trust cost basis is unknown anyway).
PROXY = {"VINIX": "VOO", "VIEIX": "VXF"}
CASH_SYMBOLS = {"SPAXX", "FDRXX", "QPCTQ", "CASH", "VMFXX"}


def main():
    df = pd.read_excel("profolio.xlsx", sheet_name="Holdings Detail")
    df = df[df["Symbol"].notna()]
    df = df[~df["Name"].astype(str).str.upper().str.contains("TOTAL")]

    # openpyxl-saved files have no cached formula values; recover MV if needed.
    mv = pd.to_numeric(df["Market Value"], errors="coerce")
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    px_sheet = pd.to_numeric(df["Current Price"], errors="coerce")
    df["mv"] = mv.fillna(qty * px_sheet)

    need_px = [str(s).upper().strip().replace("*", "") for s in df["Symbol"].unique()]
    need_px = [s for s in need_px if s not in CASH_SYMBOLS and s not in PROXY]
    need_px = sorted(set(need_px) | set(PROXY.values()))
    live = yf.download(need_px, period="5d", interval="1d", progress=False,
                       threads=False)["Close"].ffill().iloc[-1]
    live_px = {s: float(live[s]) for s in need_px}

    rows, total_cost, notes = [], 0.0, []
    actual_cash = 0.0
    for _, r in df.iterrows():
        sym = str(r["Symbol"]).upper().strip().replace("*", "")
        if sym in CASH_SYMBOLS:
            actual_cash += float(r["mv"])
            continue
        gain = pd.to_numeric(pd.Series([r.get("Unrealized Gain $")]), errors="coerce").iloc[0]
        gain = 0.0 if pd.isna(gain) else float(gain)
        cost_total = float(r["mv"]) - gain
        q = r["Quantity"]
        c = r["Avg Cost"]
        if sym in PROXY:  # 401(k) trust -> proxy ETF at equal dollar value
            target = PROXY[sym]
            lp = live_px[target]
            qty_f, fill = float(r["mv"]) / lp, lp
            cost_total = qty_f * fill
            notes.append(f"{sym} -> {target}: {qty_f:.3f} sh @ {lp:.2f} (equal-value proxy, cost basis reset)")
            sym = target
        elif pd.notna(q) and pd.notna(c):
            qty_f, fill = float(q), float(c)
        elif pd.notna(q):  # qty known, cost unknown (vested RSU)
            qty_f, fill = float(q), float(r["Current Price"])
            cost_total = qty_f * fill
            notes.append(f"{sym}: cost basis unknown, booked at current price {fill:.2f}")
        else:  # M1 rows: no qty, no cost
            lp = live_px[sym]
            qty_f = float(r["mv"]) / lp
            fill = cost_total / qty_f if qty_f else 0.0
            notes.append(f"{sym} (M1): qty inferred {qty_f:.3f} from live price {lp:.2f}")
        exch = EXCHANGE.get(sym, "NASDAQ")
        rows.append([f"{exch}:{sym}", "Buy", round(qty_f, 4), round(fill, 2), 0, TRADE_DATE])
        total_cost += qty_f * fill

    deposit = actual_cash - TV_DEFAULT_CASH + total_cost
    side = "Deposit" if deposit >= 0 else "Withdrawal"
    rows.append(["$CASH", side, round(abs(deposit), 2), 0 if side == "Deposit" else "", 0, TRADE_DATE])

    out = pd.DataFrame(rows, columns=["Symbol", "Side", "Qty", "Fill Price",
                                      "Commission", "Closing Time"])
    out.to_csv(OUT, index=False)

    tv_cash = TV_DEFAULT_CASH + deposit - total_cost
    print(f"wrote {len(out)} rows -> {OUT}")
    print(f"total buy cost : ${total_cost:,.2f}")
    print(f"cash {side.lower()}: ${abs(deposit):,.2f}")
    print(f"TV cash check  : ${tv_cash:,.2f} (target ${actual_cash:,.2f})")
    print(f"position count : {len(out) - 1}")
    for n in notes:
        print("  note:", n)


if __name__ == "__main__":
    main()
