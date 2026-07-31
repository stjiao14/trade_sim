"""Convert profolio.xlsx (multi-broker holdings workbook) into the holdings CSV
format expected by concentration_analysis.py / factor_xray.py.

Fixes applied:
- Drops TOTAL/summary rows (they double-count liquid net worth).
- Maps Symbol/Name/Market Value -> ticker_symbol/name/institution_value.
- Adds subtype column; vested RSU shares are plain stock (subtype 'rsu' is
  reserved for UNVESTED RSU by the analysis scripts).
- Strips asterisks from 401(k) institutional trust tickers (VINIX*/VIEIX*).
- Forces known cash-like rows to subtype 'cash' with name containing 'cash'
  so factor_xray.py treats them as zero-return CASH (e.g. QPCTQ FDIC deposit).

Usage: .venv/bin/python xlsx_to_holdings.py [input.xlsx] [output.csv]
"""
import sys

import pandas as pd

CASH_TICKERS = {"SPAXX", "FDRXX", "QPCTQ", "CASH", "VMFXX"}
ETF_TICKERS = {"QQQM", "QQQ", "SPY", "TQQQ", "GLD", "DRAM", "VINIX", "VIEIX"}

# Unvested RSU is not in any broker export; the share count lives in
# git-ignored config_local.py so no personal data appears in this file.
# Value is approximated as shares x reference price; concentration_analysis.py
# converts it back to shares using the live GOOGL price, so small price drift
# between REFERENCE_PRICE and the live price is harmless.
try:
    from config_local import UNVESTED_RSU_SHARES
except ImportError:
    UNVESTED_RSU_SHARES = 0.0
REFERENCE_GOOGL_PRICE = 319.0


def convert(xlsx_path="profolio.xlsx", out_path="holdings.csv"):
    df = pd.read_excel(xlsx_path, sheet_name="Holdings Detail")

    # Drop TOTAL / blank rows.
    df = df[df["Symbol"].notna()]
    df = df[~df["Name"].astype(str).str.upper().str.contains("TOTAL")]

    df["ticker_symbol"] = (
        df["Symbol"].astype(str).str.upper().str.strip().str.replace("*", "", regex=False)
    )
    # Files saved by openpyxl have no cached formula values, so a formula cell
    # (e.g. Market Value = Qty x Price) reads as NaN; fall back to Qty x Price.
    mv = pd.to_numeric(df["Market Value"], errors="coerce")
    calc = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(
        df["Current Price"], errors="coerce")
    df["institution_value"] = mv.fillna(calc).fillna(0.0)

    def subtype(row):
        tk = row["ticker_symbol"]
        if tk in CASH_TICKERS:
            return "cash"
        if tk in ETF_TICKERS:
            return "etf"
        return "stock"

    df["subtype"] = df.apply(subtype, axis=1)

    # Ensure cash-like names contain a keyword factor_xray.py recognizes.
    def name(row):
        n = str(row["Name"])
        if row["subtype"] == "cash" and not any(
            k in n.lower() for k in ["cash", "money market", "sweep"]
        ):
            return n + " (Cash)"
        return n

    df["name"] = df.apply(name, axis=1)

    out = df[["Account", "ticker_symbol", "name", "subtype", "institution_value"]]
    out = out.rename(columns={"Account": "account"})

    if UNVESTED_RSU_SHARES > 0:
        rsu_row = pd.DataFrame([{
            "account": "Google RSU (unvested)",
            "ticker_symbol": "GOOGL",
            "name": "Alphabet Unvested RSU",
            "subtype": "rsu",
            "institution_value": UNVESTED_RSU_SHARES * REFERENCE_GOOGL_PRICE,
        }])
        out = pd.concat([out, rsu_row], ignore_index=True)

    out.to_csv(out_path, index=False)

    total = out["institution_value"].sum()
    print(f"Wrote {len(out)} rows -> {out_path} | total market value: ${total:,.2f}")
    print(out.groupby("subtype")["institution_value"].agg(["count", "sum"]).to_string())
    return out


if __name__ == "__main__":
    convert(
        sys.argv[1] if len(sys.argv) > 1 else "profolio.xlsx",
        sys.argv[2] if len(sys.argv) > 2 else "holdings.csv",
    )
