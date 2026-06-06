"""Whole-portfolio factor x-ray plus true diversification score.

Real holdings are read from a local CSV. Do not hardcode or commit real dollar amounts.
This script is not investment advice.
"""
import sys

import numpy as np
import pandas as pd
import yfinance as yf

from concentration_analysis import _name, _num, _ticker


FACTOR_PROXIES = {"MKT": "SPY", "TECH": "XLK", "VALUE": "IWD", "SIZE": "IWM",
                  "INTL": "EFA", "RATES": "IEF", "GOLD": "GLD", "OIL": "USO"}


def _subtype(row):
    v = row.get("subtype", "")
    return "" if pd.isna(v) else str(v).lower().strip()


def _add(weights, ticker, usd):
    weights[ticker] = weights.get(ticker, 0.0) + float(usd)


def load_full_portfolio(csv_path):
    """Return {ticker_for_yf: usd_value} from a holdings CSV.

    GOOG/GOOGL/RSU are merged into GOOGL; cash/money-market rows become CASH;
    rows without usable tickers are reported in notes.
    """
    df = pd.read_csv(csv_path)
    required = {"ticker_symbol", "name", "subtype", "institution_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    weights = {}
    notes = []
    for _, row in df.iterrows():
        val = _num(row.get("institution_value", 0.0))
        if val == 0:
            continue
        tk = _ticker(row)
        nm = _name(row)
        st = _subtype(row)
        nm_l = nm.lower()

        if tk in {"GOOG", "GOOGL"}:
            _add(weights, "GOOGL", val)
            if st == "rsu":
                notes.append(f"RSU merged into GOOGL: {val:,.0f} USD")
            continue

        if (not tk) and "500 index" in nm_l:
            _add(weights, "SPY", val)
            notes.append(f"No-ticker 500 index trust proxied by SPY: {val:,.0f} USD")
            continue

        if (not tk) and ("ext market" in nm_l or "extended market" in nm_l):
            _add(weights, "VXF", val)
            notes.append(f"No-ticker extended market index trust proxied by VXF: {val:,.0f} USD")
            continue

        # Cash, money market, sweep, and short-term reserve rows are zero-return CASH.
        if tk in {"CASH", "USD"} or tk.startswith("CUR:") or any(
            k in nm_l for k in ["cash", "money market", "sweep", "core position"]
        ):
            _add(weights, "CASH", val)
            continue

        if tk:
            _add(weights, tk, val)
        else:
            notes.append(f"Skipped no-ticker row: {nm or '(blank name)'} {val:,.0f} USD")

    if not weights:
        raise ValueError("CSV has no usable holdings")
    return weights, notes


def _download_close(tickers, lookback_days):
    px = yf.download(tickers, period=f"{lookback_days}d", interval="1d",
                     progress=False, auto_adjust=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px


def portfolio_returns(weights_usd, lookback_days=750):
    """Download daily returns and build a USD-weighted portfolio return series."""
    tickers = [t for t in weights_usd if t != "CASH"]
    if len(tickers) < 2 and "CASH" not in weights_usd:
        raise ValueError("need at least 2 return-bearing holdings")
    if tickers:
        px = _download_close(tickers, lookback_days)
        R = px.pct_change().dropna(how="all")
        R = R.dropna(axis=1, how="all").fillna(0.0)
    else:
        R = pd.DataFrame(index=pd.RangeIndex(lookback_days))
    if "CASH" in weights_usd:
        R = R.assign(CASH=0.0)
    cols = list(R.columns)
    live_cols = [c for c in cols if c != "CASH"]
    if len(live_cols) < 2:
        raise ValueError("need at least 2 return-bearing holdings; CASH is zero-vol")
    w = np.array([weights_usd[c] for c in cols], float)
    w = w / w.sum()
    port = pd.Series(R[cols].values @ w, index=R.index, name="portfolio")
    return R[cols], w, port


def diversification(R, w):
    """Compute ENB, diversification ratio, PC1 share, and risk contribution table."""
    w = np.asarray(w, float)
    w = w / w.sum()
    Sig = R.cov().values
    sig = np.sqrt(np.diag(Sig))
    var_p = float(w @ Sig @ w)
    if var_p <= 0:
        raise ValueError("portfolio volatility is zero; cannot compute diversification")
    rc = w * (Sig @ w) / var_p
    C = R.corr().fillna(0.0).values.copy()
    np.fill_diagonal(C, 1.0)
    lamC = np.linalg.eigvalsh(C)
    ENB = float(lamC.sum()**2 / (lamC**2).sum())
    DR = float((w @ sig) / np.sqrt(var_p))
    PC1 = float(lamC.max() / lamC.sum())
    risk_contrib = dict(sorted(zip(R.columns, (rc * 100).round(1)), key=lambda kv: -kv[1]))
    return dict(ENB=ENB, DR=DR, PC1_share=PC1, n_holdings=len(w), risk_contrib_pct=risk_contrib)


def factor_xray(port, lookback_days=750, proxies=FACTOR_PROXIES):
    """Run an ETF-proxy OLS factor x-ray. Proxies are collinear; read R2 and dominant betas."""
    px = _download_close(list(proxies.values()), lookback_days)
    fx = px.pct_change().dropna()
    fx.columns = list(proxies.keys())
    df = pd.concat([port, fx], axis=1, join="inner").dropna()
    if len(df) <= len(proxies) + 2:
        raise ValueError("return history is too short for factor regression")
    y = df["portfolio"].values
    X = np.column_stack([np.ones(len(df)), df[list(proxies.keys())].values])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag((resid @ resid) / dof * np.linalg.pinv(X.T @ X)))
    r2 = 1 - (resid @ resid) / (((y - y.mean())**2).sum())
    tab = pd.DataFrame({"beta": b, "t": b / se}, index=["alpha"] + list(proxies.keys()))
    cond = float(np.linalg.cond(X))
    return tab, float(r2), cond


def _csv_path_from_config():
    try:
        import config_local as C
    except ImportError:
        return None
    return getattr(C, "HOLDINGS_CSV", None) or getattr(C, "CSV_PATH", None)


def print_report(csv_path):
    """Print factor table, diversification score, and risk contribution."""
    weights, notes = load_full_portfolio(csv_path)
    R, w, port = portfolio_returns(weights)
    tab, r2, cond = factor_xray(port)
    div = diversification(R, w)
    usd_total = sum(weights.values())
    usd_pct = {k: weights[k] / usd_total * 100 for k in weights}

    print("== Whole-Portfolio Factor X-ray ==")
    if notes:
        print("Load notes:")
        for n in notes:
            print(f"- {n}")
    print("\nFactor regression (beta / t):")
    print(tab.round(3).to_string())
    print(f"R²: {r2:.2%} | condition number: {cond:.1f}")
    if cond > 30:
        print("Note: factor proxies are collinear; read R2 and dominant betas, not each beta as orthogonal exposure.")

    print("\n== True Diversification ==")
    print(f"You nominally hold {div['n_holdings']} assets, but only about {div['ENB']:.1f} independent bets; "
          f"one common factor (PC1) explains {div['PC1_share']:.1%} of cross-sectional variance; "
          f"diversification ratio DR={div['DR']:.2f} (1.0=no diversification benefit).")

    print("\nRisk contribution (% of portfolio variance):")
    for tk, rc in div["risk_contrib_pct"].items():
        print(f"{tk:>8s}: risk {rc:5.1f}% | dollars {usd_pct.get(tk, 0.0):5.1f}%")
    top = next(iter(div["risk_contrib_pct"]))
    print(f"\nLargest risk source: {top} contributes {div['risk_contrib_pct'][top]:.1f}% of risk, "
          f"but only {usd_pct.get(top, 0.0):.1f}% of dollars.")
    print("\nNotes: CASH is zero-vol; RSU is merged into GOOGL; need at least 2 return-bearing assets; this script is not investment advice.")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _csv_path_from_config()
    if not csv_path:
        print("ERROR: pass a local holdings CSV path or set HOLDINGS_CSV/CSV_PATH in config_local.py.")
        return
    try:
        print_report(csv_path)
    except Exception as exc:
        print(f"ERROR: factor_xray failed: {exc}")


if __name__ == "__main__":
    main()
