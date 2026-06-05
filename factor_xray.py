"""全组合因子 X 光 + 真分散度评分。

真实持仓从本地 CSV 读取;不要把真实金额写进代码或提交到仓库。本脚本不构成投资建议。
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
    """从持仓 CSV 返回 {ticker_for_yf: usd_value}(每个标的独立)。
    GOOG/GOOGL/RSU 合并到 GOOGL;现金/货币基金记 CASH;无 ticker 行会写入 notes。"""
    df = pd.read_csv(csv_path)
    required = {"ticker_symbol", "name", "subtype", "institution_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

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
                notes.append(f"RSU 并入 GOOGL: {val:,.0f} USD")
            continue

        if (not tk) and "500 index" in nm_l:
            _add(weights, "SPY", val)
            notes.append(f"无 ticker 500 index trust 用 SPY 代理: {val:,.0f} USD")
            continue

        if (not tk) and ("ext market" in nm_l or "extended market" in nm_l):
            _add(weights, "VXF", val)
            notes.append(f"无 ticker extended market index trust 用 VXF 代理: {val:,.0f} USD")
            continue

        # 现金、money market、sweep、短期储备按零收益 CASH 处理。
        if tk in {"CASH", "USD"} or tk.startswith("CUR:") or any(
            k in nm_l for k in ["cash", "money market", "sweep", "core position"]
        ):
            _add(weights, "CASH", val)
            continue

        if tk:
            _add(weights, tk, val)
        else:
            notes.append(f"跳过无 ticker 行: {nm or '(blank name)'} {val:,.0f} USD")

    if not weights:
        raise ValueError("CSV 没有可用持仓")
    return weights, notes


def _download_close(tickers, lookback_days):
    px = yf.download(tickers, period=f"{lookback_days}d", interval="1d",
                     progress=False, auto_adjust=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(tickers[0])
    return px


def portfolio_returns(weights_usd, lookback_days=750):
    """下载各标的日收益,按美元权重合成组合收益。CASH 视作 0 收益、0 波动。"""
    tickers = [t for t in weights_usd if t != "CASH"]
    if len(tickers) < 2 and "CASH" not in weights_usd:
        raise ValueError("需至少 2 个有收益历史的标的")
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
        raise ValueError("需至少 2 个有收益历史的标的;CASH 按零波动处理")
    w = np.array([weights_usd[c] for c in cols], float)
    w = w / w.sum()
    port = pd.Series(R[cols].values @ w, index=R.index, name="portfolio")
    return R[cols], w, port


def diversification(R, w):
    """ENB(独立押注) / DR(分散比) / PC1占比 / 风险贡献表。"""
    w = np.asarray(w, float)
    w = w / w.sum()
    Sig = R.cov().values
    sig = np.sqrt(np.diag(Sig))
    var_p = float(w @ Sig @ w)
    if var_p <= 0:
        raise ValueError("组合波动为 0,无法计算分散度")
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
    """用 ETF 代理做 OLS 因子 X 光。代理彼此共线,重读 R2 与主导 beta。"""
    px = _download_close(list(proxies.values()), lookback_days)
    fx = px.pct_change().dropna()
    fx.columns = list(proxies.keys())
    df = pd.concat([port, fx], axis=1, join="inner").dropna()
    if len(df) <= len(proxies) + 2:
        raise ValueError("收益历史太短,无法做因子回归")
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
    """打印因子表、真分散度和风险贡献。"""
    weights, notes = load_full_portfolio(csv_path)
    R, w, port = portfolio_returns(weights)
    tab, r2, cond = factor_xray(port)
    div = diversification(R, w)
    usd_total = sum(weights.values())
    usd_pct = {k: weights[k] / usd_total * 100 for k in weights}

    print("== 全组合因子 X 光 ==")
    if notes:
        print("读取备注:")
        for n in notes:
            print(f"- {n}")
    print("\n因子回归(beta / t):")
    print(tab.round(3).to_string())
    print(f"R²: {r2:.2%} | condition number: {cond:.1f}")
    if cond > 30:
        print("提示:因子代理有共线性;重读 R² 与主导 beta,不要把每个 beta 当正交暴露。")

    print("\n== 真分散度 ==")
    print(f"你名义持有 {div['n_holdings']} 个标的,实际约 {div['ENB']:.1f} 个独立押注; "
          f"一个共同因子(PC1)解释了 {div['PC1_share']:.1%} 的横截面方差; "
          f"分散比 DR={div['DR']:.2f}(1.0=无分散收益)。")

    print("\n风险贡献(组合方差占比):")
    for tk, rc in div["risk_contrib_pct"].items():
        print(f"{tk:>8s}: 风险 {rc:5.1f}% | 美元 {usd_pct.get(tk, 0.0):5.1f}%")
    top = next(iter(div["risk_contrib_pct"]))
    print(f"\n最大风险源: {top} 占风险 {div['risk_contrib_pct'][top]:.1f}%, "
          f"但美元占比 {usd_pct.get(top, 0.0):.1f}%。")
    print("\n注意:CASH 按零波动;RSU 并入 GOOGL;需至少 2 个有收益历史的标的;本脚本不构成投资建议。")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else _csv_path_from_config()
    if not csv_path:
        print("ERROR: 请传入本地持仓 CSV 路径,或在 config_local.py 设置 HOLDINGS_CSV/CSV_PATH。")
        return
    try:
        print_report(csv_path)
    except Exception as exc:
        print(f"ERROR: factor_xray 失败: {exc}")


if __name__ == "__main__":
    main()
