import sys

import numpy as np
import pandas as pd
import yfinance as yf

TICKERS = ["GOOGL", "SPY", "QQQ", "TQQQ"]

try:
    from config_local import FALLBACK_WEIGHTS, HC_YEARS, HOLDINGS, SALARY_USD
    CONFIG_SOURCE = "config_local.py"
except ImportError:
    from config_example import FALLBACK_WEIGHTS, HC_YEARS, HOLDINGS, SALARY_USD
    CONFIG_SOURCE = "config_example.py"


def fetch_prices(tickers=TICKERS):
    """Latest close prices."""
    px = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"].iloc[-1]
    return {t: float(px[t]) for t in tickers}


def fetch_alphabet_weight(etf, fallback):
    """Alphabet weight inside an ETF, GOOGL+GOOG combined. Prefer yfinance, fallback loudly."""
    try:
        th = yf.Ticker(etf).funds_data.top_holdings   # DataFrame index=symbol, col 'Holding Percent'
        w = float(th.loc[th.index.isin(["GOOGL", "GOOG"]), "Holding Percent"].sum())
        if w > 0:
            return w, "fetched"
    except Exception:
        pass
    return float(fallback), "FALLBACK(please verify manually)"


def _num(x):
    v = pd.to_numeric(x, errors="coerce")
    return 0.0 if pd.isna(v) else float(v)


def _ticker(row):
    v = row.get("ticker_symbol", "")
    return "" if pd.isna(v) else str(v).upper().strip()


def _name(row):
    v = row.get("name", "")
    return "" if pd.isna(v) else str(v)


def load_holdings_from_csv(path, px):
    """Build the HOLDINGS dict from a local holdings CSV. Real dollars stay local."""
    df = pd.read_csv(path)
    required = {"ticker_symbol", "name", "subtype", "institution_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    H = dict(googl_shares=0.0, rsu_unvested_shares=0.0,
             spy_usd=0.0, qqq_usd=0.0, tqqq_usd=0.0)
    notes = []
    liquid_total = 0.0

    for _, row in df.iterrows():
        val = _num(row.get("institution_value", 0.0))
        if val == 0:
            continue
        tk = _ticker(row)
        nm = _name(row)
        subtype = "" if pd.isna(row.get("subtype", "")) else str(row.get("subtype", "")).lower()
        is_rsu = subtype == "rsu"

        if not is_rsu:
            liquid_total += val

        # Convert GOOG/GOOGL dollar value into GOOGL-equivalent shares to avoid A/C share price noise.
        if tk in {"GOOGL", "GOOG"}:
            if is_rsu:
                H["rsu_unvested_shares"] += val / px["GOOGL"]
                notes.append(f"RSU: {tk} {val:,.0f} USD -> GOOGL equivalent")
            else:
                H["googl_shares"] += val / px["GOOGL"]
                notes.append(f"Direct Alphabet: {tk} {val:,.0f} USD -> GOOGL equivalent")
            continue

        if tk == "SPY" or "500 Index" in nm:
            H["spy_usd"] += val
            notes.append(f"SPY-like: {tk or nm} {val:,.0f} USD")
        elif tk in {"QQQ", "QQQM"} or "NASDAQ 100" in nm.upper():
            H["qqq_usd"] += val
            notes.append(f"QQQ-like: {tk or nm} {val:,.0f} USD")
        elif tk == "TQQQ":
            H["tqqq_usd"] += val
            notes.append(f"TQQQ: {val:,.0f} USD")

    H["total_liquid_usd"] = liquid_total
    return H, notes, df


def compute_lookthrough(H, px, w_spy, w_qqq):
    googl_stock = H["googl_shares"] * px["GOOGL"]
    rsu         = H["rsu_unvested_shares"] * px["GOOGL"]
    spy_lt  = H["spy_usd"]  * w_spy
    qqq_lt  = H["qqq_usd"]  * w_qqq
    tqqq_lt = H["tqqq_usd"] * w_qqq * 3          # TQQQ 3x notional exposure; path decay is separate.
    rows = [("GOOGL stock", googl_stock, googl_stock),
            ("SPY look-through",  H["spy_usd"],  spy_lt),
            ("QQQ look-through",  H["qqq_usd"],  qqq_lt),
            ("TQQQ look-through(3x notional)", H["tqqq_usd"], tqqq_lt)]
    liquid_googl = googl_stock + spy_lt + qqq_lt + tqqq_lt
    liquid_nw    = H.get("total_liquid_usd", googl_stock + H["spy_usd"] + H["qqq_usd"] + H["tqqq_usd"])
    return rows, liquid_googl, liquid_nw, rsu


def corr_and_beta(H, px, lookback_days=750):
    """Estimate beta/R2 of the GOOGL/SPY/QQQ/TQQQ sleeve to GOOGL from daily returns."""
    hist = yf.download(TICKERS, period=f"{lookback_days}d", interval="1d", progress=False)["Close"]
    R = hist.pct_change().dropna()
    usd = np.array([H["googl_shares"]*px["GOOGL"], H["spy_usd"], H["qqq_usd"], H["tqqq_usd"]], float)
    if usd.sum() <= 0:
        raise ValueError("liquid holdings are zero; cannot compute portfolio beta/R2")
    wts = usd / usd.sum()
    port = R[TICKERS].values @ wts
    g = R["GOOGL"].values
    beta = float(np.cov(port, g)[0, 1] / np.var(g))
    r2 = float(np.corrcoef(port, g)[0, 1] ** 2)
    return R.corr(), beta, r2


def _usd(x):
    return f"${x:,.0f}"


def _pct(x):
    return f"{x*100:,.1f}%"


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def _csv_path_from_config():
    try:
        import config_local as C
    except ImportError:
        return None
    return getattr(C, "HOLDINGS_CSV", None) or getattr(C, "CSV_PATH", None)


def print_report(csv_path=None):
    csv_path = csv_path or _csv_path_from_config()
    print("=== 1) Prices and Alphabet weights inside ETFs ===")
    px = fetch_prices()
    if csv_path:
        H, notes, _ = load_holdings_from_csv(csv_path, px)
        print(f"Holdings source: local CSV {csv_path}")
    else:
        H = HOLDINGS
        notes = []
        if CONFIG_SOURCE != "config_local.py":
            print("WARNING: config_local.py not found; using dummy values from config_example.py.")
            print("         Copy config_example.py -> config_local.py and fill real holdings; config_local.py is git-ignored.")
        print(f"Holdings source: {CONFIG_SOURCE}")

    w_spy, src_spy = fetch_alphabet_weight("SPY", FALLBACK_WEIGHTS["spy_alphabet"])
    w_qqq, src_qqq = fetch_alphabet_weight("QQQ", FALLBACK_WEIGHTS["qqq_alphabet"])
    for t in TICKERS:
        print(f"{t:5s}: {_usd(px[t])}")
    print(f"SPY Alphabet(GOOGL+GOOG) weight: {_pct(w_spy)} [{src_spy}]")
    print(f"QQQ Alphabet(GOOGL+GOOG) weight: {_pct(w_qqq)} [{src_qqq}]")
    if "FALLBACK" in src_spy or "FALLBACK" in src_qqq:
        print("!!! FALLBACK: ETF weights are manual fallback values; verify issuer holdings before concluding.")
    if notes:
        print("\nCSV mapping summary:")
        for n in notes:
            print(f"- {n}")

    rows, liquid_googl, liquid_nw, rsu = compute_lookthrough(H, px, w_spy, w_qqq)
    table = pd.DataFrame(rows, columns=["source", "market_value", "googl_equivalent_exposure"])
    print("\n=== 2) Look-through exposure table ===")
    print(table.assign(market_value=table["market_value"].map(_usd),
                       googl_equivalent_exposure=table["googl_equivalent_exposure"].map(_usd)).to_string(index=False))

    hc = float(SALARY_USD) * float(HC_YEARS)
    with_rsu = liquid_nw + rsu
    with_hc = with_rsu + hc
    print("\n=== 3) GOOGL-equivalent exposure share ===")
    print(f"Liquid account, ex-RSU/income: {_usd(liquid_googl)} / {_usd(liquid_nw)} = {_pct(_safe_div(liquid_googl, liquid_nw))}")
    print(f"Including unvested RSU:        {_usd(liquid_googl + rsu)} / {_usd(with_rsu)} = {_pct(_safe_div(liquid_googl + rsu, with_rsu))}")
    if HC_YEARS and SALARY_USD:
        print(f"Including human capital:       {_usd(liquid_googl + rsu + hc)} / {_usd(with_hc)} = {_pct(_safe_div(liquid_googl + rsu + hc, with_hc))}")
    else:
        print("Including human capital:       disabled (SALARY_USD or HC_YEARS is 0)")

    print("\n=== 4) GOOGL shock scenarios ===")
    for shock in (-0.30, -0.50):
        liquid_loss = liquid_googl * shock
        rsu_loss = (liquid_googl + rsu) * shock
        print(f"GOOGL {shock:+.0%}: liquid-account loss {_usd(liquid_loss)} | including-RSU loss {_usd(rsu_loss)}")
    print("Note: a GOOGL drawdown may also affect job stability and future RSU vesting; this is not ordinary stock beta.")

    print("\n=== 5) GOOGL/SPY/QQQ/TQQQ sleeve beta/R2 to GOOGL ===")
    try:
        corr, beta, r2 = corr_and_beta(H, px)
        print(corr.round(2).to_string())
        print(f"\nSleeve beta to GOOGL: {beta:.2f} | R2: {r2:.2f}")
        print("Read this as the GOOGL/SPY/QQQ/TQQQ tech/index sleeve only, not a whole-account factor regression; use factor_xray.py for the whole book.")
    except Exception as e:
        print(f"Correlation/beta fetch failed: {e}")

    print("\nCaveats:")
    print("- GOOG + GOOGL are combined as Alphabet exposure.")
    print("- TQQQ uses 3x notional exposure; path decay, daily reset, and vol drag are separate.")
    print("- Human capital is illustrative, not a tradable asset valuation.")
    print("- beta/R2 only covers the GOOGL/SPY/QQQ/TQQQ sleeve; use factor_xray.py for whole-book risk.")
    print("- This script is not investment advice.")


if __name__=="__main__":
    print_report(sys.argv[1] if len(sys.argv) > 1 else None)
