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
    """最新收盘价。"""
    px = yf.download(tickers, period="5d", interval="1d", progress=False)["Close"].iloc[-1]
    return {t: float(px[t]) for t in tickers}


def fetch_alphabet_weight(etf, fallback):
    """ETF 内 GOOGL+GOOG 合计权重。优先 yfinance funds_data,失败用兜底并告警。"""
    try:
        th = yf.Ticker(etf).funds_data.top_holdings   # DataFrame index=symbol, col 'Holding Percent'
        w = float(th.loc[th.index.isin(["GOOGL", "GOOG"]), "Holding Percent"].sum())
        if w > 0:
            return w, "fetched"
    except Exception:
        pass
    return float(fallback), "FALLBACK(请手动核对!)"


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
    """从持仓 CSV 直接生成脚本所需 HOLDINGS。真实金额只留在本地 CSV,不写入代码。"""
    df = pd.read_csv(path)
    required = {"ticker_symbol", "name", "subtype", "institution_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要列: {sorted(missing)}")

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

        # GOOG/GOOGL 统一按美元价值折算成 GOOGL 等价股数,避免 A/C 股价格小差异污染敞口。
        if tk in {"GOOGL", "GOOG"}:
            if is_rsu:
                H["rsu_unvested_shares"] += val / px["GOOGL"]
                notes.append(f"RSU: {tk} {val:,.0f} USD -> GOOGL 等价")
            else:
                H["googl_shares"] += val / px["GOOGL"]
                notes.append(f"直接 Alphabet: {tk} {val:,.0f} USD -> GOOGL 等价")
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
    tqqq_lt = H["tqqq_usd"] * w_qqq * 3          # TQQQ 3x 名义敞口(路径衰减是另一个分析)
    rows = [("GOOGL 股票", googl_stock, googl_stock),
            ("SPY look-through",  H["spy_usd"],  spy_lt),
            ("QQQ look-through",  H["qqq_usd"],  qqq_lt),
            ("TQQQ look-through(3x名义)", H["tqqq_usd"], tqqq_lt)]
    liquid_googl = googl_stock + spy_lt + qqq_lt + tqqq_lt
    liquid_nw    = H.get("total_liquid_usd", googl_stock + H["spy_usd"] + H["qqq_usd"] + H["tqqq_usd"])
    return rows, liquid_googl, liquid_nw, rsu


def corr_and_beta(H, px, lookback_days=750):
    """近 ~3 年日收益:GOOGL/SPY/QQQ/TQQQ sleeve 对 GOOGL 的 beta 与 R²。"""
    hist = yf.download(TICKERS, period=f"{lookback_days}d", interval="1d", progress=False)["Close"]
    R = hist.pct_change().dropna()
    usd = np.array([H["googl_shares"]*px["GOOGL"], H["spy_usd"], H["qqq_usd"], H["tqqq_usd"]], float)
    if usd.sum() <= 0:
        raise ValueError("流动持仓为 0,无法计算组合 beta/R²")
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
    print("=== 1) 价格与 ETF 内 Alphabet 权重 ===")
    px = fetch_prices()
    if csv_path:
        H, notes, _ = load_holdings_from_csv(csv_path, px)
        print(f"持仓来源: CSV 本地文件 {csv_path}")
    else:
        H = HOLDINGS
        notes = []
        if CONFIG_SOURCE != "config_local.py":
            print("WARNING: 未找到 config_local.py, 正在使用 config_example.py 的 dummy 数字。")
            print("         请复制 config_example.py -> config_local.py 并填入真实持仓;config_local.py 已被 git-ignore。")
        print(f"持仓来源: {CONFIG_SOURCE}")

    w_spy, src_spy = fetch_alphabet_weight("SPY", FALLBACK_WEIGHTS["spy_alphabet"])
    w_qqq, src_qqq = fetch_alphabet_weight("QQQ", FALLBACK_WEIGHTS["qqq_alphabet"])
    for t in TICKERS:
        print(f"{t:5s}: {_usd(px[t])}")
    print(f"SPY Alphabet(GOOGL+GOOG) 权重: {_pct(w_spy)} [{src_spy}]")
    print(f"QQQ Alphabet(GOOGL+GOOG) 权重: {_pct(w_qqq)} [{src_qqq}]")
    if "FALLBACK" in src_spy or "FALLBACK" in src_qqq:
        print("!!! FALLBACK: ETF 权重为手动兜底值,请去发行商持仓页核对后再作结论。")
    if notes:
        print("\nCSV 映射摘要:")
        for n in notes:
            print(f"- {n}")

    rows, liquid_googl, liquid_nw, rsu = compute_lookthrough(H, px, w_spy, w_qqq)
    table = pd.DataFrame(rows, columns=["来源", "市值", "GOOGL等价敞口"])
    print("\n=== 2) 穿透敞口表 ===")
    print(table.assign(市值=table["市值"].map(_usd),
                       GOOGL等价敞口=table["GOOGL等价敞口"].map(_usd)).to_string(index=False))

    hc = float(SALARY_USD) * float(HC_YEARS)
    with_rsu = liquid_nw + rsu
    with_hc = with_rsu + hc
    print("\n=== 3) GOOGL 等价敞口占比 ===")
    print(f"流动账户(不含 RSU/收入): {_usd(liquid_googl)} / {_usd(liquid_nw)} = {_pct(_safe_div(liquid_googl, liquid_nw))}")
    print(f"含未归属 RSU:          {_usd(liquid_googl + rsu)} / {_usd(with_rsu)} = {_pct(_safe_div(liquid_googl + rsu, with_rsu))}")
    if HC_YEARS and SALARY_USD:
        print(f"含人力资本(示意):      {_usd(liquid_googl + rsu + hc)} / {_usd(with_hc)} = {_pct(_safe_div(liquid_googl + rsu + hc, with_hc))}")
    else:
        print("含人力资本(示意):      已关闭(SALARY_USD 或 HC_YEARS 为 0)")

    print("\n=== 4) GOOGL 冲击情景 ===")
    for shock in (-0.30, -0.50):
        liquid_loss = liquid_googl * shock
        rsu_loss = (liquid_googl + rsu) * shock
        print(f"GOOGL {shock:+.0%}: 流动账户损失 {_usd(liquid_loss)} | 含 RSU 损失 {_usd(rsu_loss)}")
    print("提示:GOOGL 大跌也可能同时打击工作稳定性与未来 RSU 归属,这不是普通股票 beta。")

    print("\n=== 5) GOOGL/SPY/QQQ/TQQQ sleeve 对 GOOGL 的 beta/R² ===")
    try:
        corr, beta, r2 = corr_and_beta(H, px)
        print(corr.round(2).to_string())
        print(f"\n该 sleeve 对 GOOGL 的 beta: {beta:.2f} | R²: {r2:.2f}")
        print("读法:这里只覆盖 GOOGL/SPY/QQQ/TQQQ 这条科技/指数 sleeve,不是全账户因子回归;全账户看 factor_xray.py。")
    except Exception as e:
        print(f"相关/beta 获取失败: {e}")

    print("\nCaveats:")
    print("- GOOG + GOOGL 合并计算 Alphabet 敞口。")
    print("- TQQQ 用 3x 名义敞口;路径衰减、日重置、波动损耗另算。")
    print("- 人力资本一行仅为示意,不是可交易资产估值。")
    print("- beta/R² 只覆盖 GOOGL/SPY/QQQ/TQQQ sleeve;全组合风险请看 factor_xray.py。")
    print("- 本脚本不构成投资建议。")


if __name__=="__main__":
    print_report(sys.argv[1] if len(sys.argv) > 1 else None)
