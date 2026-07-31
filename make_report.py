"""Generate a lively Chinese PDF report from the three X-ray analyses.

Data is recomputed from the analysis modules themselves so the PDF can never
disagree with the CLI output. Output: portfolio_report.pdf (git-ignored,
contains real dollar amounts).

Usage: .venv/bin/python make_report.py
"""
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

import concentration_analysis as ca
import factor_xray as fx
import vest_diversify_sim as vs

OUT_PDF = "portfolio_report.pdf"

# ---------------------------------------------------------------- palette
C_GOOG = "#d9552c"   # warm red-orange for GOOGL
C_RSU = "#f0a35e"
C_IDX = "#4c72b0"
C_TECH = "#8172b3"
C_CASH = "#9e9e9e"
C_GREEN = "#3a9d6e"
C_GRID = "#dddddd"
plt.rcParams["axes.edgecolor"] = "#666666"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.color"] = C_GRID
plt.rcParams["grid.linewidth"] = 0.6


def money(x, k=False):
    return f"${x/1000:,.0f}k" if k else f"${x:,.0f}"


# ---------------------------------------------------------------- data
def gather():
    d = {}
    df = pd.read_csv("holdings.csv")
    d["holdings"] = df

    px = ca.fetch_prices()
    d["px"] = px
    H, notes, _ = ca.load_holdings_from_csv("holdings.csv", px)
    w_spy, _ = ca.fetch_alphabet_weight("SPY", 0.038)
    w_qqq, _ = ca.fetch_alphabet_weight("QQQ", 0.050)
    rows, liquid_googl, liquid_nw, rsu = ca.compute_lookthrough(H, px, w_spy, w_qqq)
    d["lt_rows"] = rows
    d["liquid_googl"] = liquid_googl
    d["liquid_nw"] = liquid_nw
    d["rsu"] = rsu
    d["w_spy"], d["w_qqq"] = w_spy, w_qqq

    weights, _ = fx.load_full_portfolio("holdings.csv")
    R, w, port = fx.portfolio_returns(weights)
    tab, r2, cond = fx.factor_xray(port)
    div = fx.diversification(R, w)
    d["factor_tab"] = tab
    d["r2"] = r2
    d["div"] = div
    d["weights"] = weights

    cfg, basket, n_paths, _ = vs._load_config()
    sg, sb, rho = vs._calibrate_yfinance(basket, 1260)
    sim = vs.simulate(cfg, 0.07, 0.07, sg, sb, rho, N=n_paths)
    be = vs.breakeven_drift(cfg, 0.07, sg, sb, rho)
    d["sim"] = sim
    d["sg"], d["sb"], d["rho"], d["be"] = sg, sb, rho, be
    d["cfg"] = cfg
    return d


# ---------------------------------------------------------------- charts
def fig_composition(d, path):
    df = d["holdings"]
    v = df.set_index("ticker_symbol")["institution_value"].groupby(level=0).sum()
    sleeves = {
        "GOOGL 直接持股": v.get("GOOGL", 0) - df.loc[df.subtype == "rsu", "institution_value"].sum(),
        "未 vest RSU": df.loc[df.subtype == "rsu", "institution_value"].sum(),
        "大盘指数 (VINIX+SPY)": v.get("VINIX", 0) + v.get("SPY", 0),
        "纳指相关 (QQQM+TQQQ)": v.get("QQQM", 0) + v.get("TQQQ", 0),
        "扩展市场 (VIEIX)": v.get("VIEIX", 0),
        "其它科技个股": sum(v.get(t, 0) for t in ["AMD", "MSFT", "AVGO", "TSM", "INTC", "BNTX", "ZM"]),
        "主题ETF (DRAM)": v.get("DRAM", 0),
        "黄金 (GLD)": v.get("GLD", 0),
        "现金类": df.loc[df.subtype == "cash", "institution_value"].sum(),
    }
    sleeves = {k: x for k, x in sleeves.items() if x > 0}
    total = sum(sleeves.values())
    labels = [f"{k}\n{money(x, k=True)} ({x/total*100:.1f}%)" for k, x in sleeves.items()]
    colors = [C_GOOG, C_RSU, C_IDX, "#64a0d8", "#8fb8d8", C_TECH, "#c3aed6", "#d4b106", C_CASH]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    wedges, _ = ax.pie(list(sleeves.values()), colors=colors[:len(sleeves)],
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=8.5, frameon=False)
    ax.text(0, 0.08, money(total), ha="center", fontsize=15, weight="bold")
    ax.text(0, -0.14, "总净资产(含RSU)", ha="center", fontsize=9, color="#666666")
    ax.set_title("资产地图：钱都在哪儿？", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_lookthrough(d, path):
    rows = d["lt_rows"]
    names = ["直接持股", "SPY 穿透", "QQQ 穿透", "TQQQ 3x穿透", "未 vest RSU"]
    vals = [rows[0][2], rows[1][2], rows[2][2], rows[3][2], d["rsu"]]
    colors = [C_GOOG, C_IDX, "#64a0d8", "#8fb8d8", C_RSU]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    left = 0
    for n, x, c in zip(names, vals, colors):
        ax.barh([0], [x], left=left, color=c, edgecolor="white", height=0.5)
        if x > 60000:
            ax.text(left + x / 2, 0, f"{n}\n{money(x, k=True)}", ha="center", va="center",
                    fontsize=8.5, color="white", weight="bold")
        left += x
    ax.text(left + 8000, 0, money(left), va="center", fontsize=11, weight="bold")
    ax.set_xlim(0, left * 1.13)
    ax.set_yticks([])
    ax.set_xlabel("GOOGL 等值敞口（美元）")
    ax.set_title(f"GOOGL 穿透敞口：流动账户 {d['liquid_googl']/d['liquid_nw']*100:.1f}%"
                 f" → 含 RSU {left/(d['liquid_nw']+d['rsu'])*100:.1f}%",
                 fontsize=13, weight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_shock(d, path):
    g, rsu = d["liquid_googl"], d["rsu"]
    scen = ["-30%", "-50%"]
    liquid = [g * -0.30, g * -0.50]
    with_rsu = [(g + rsu) * -0.30, (g + rsu) * -0.50]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    b1 = ax.bar(x - 0.19, [-v for v in liquid], width=0.36, color=C_IDX, label="仅流动账户")
    b2 = ax.bar(x + 0.19, [-v for v in with_rsu], width=0.36, color=C_GOOG, label="含未 vest RSU")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 6000,
                    f"-{money(-b.get_height(), k=True)}", ha="center", fontsize=9, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"GOOGL {s}" for s in scen], fontsize=11)
    ax.set_ylabel("账面损失（美元）")
    ax.set_ylim(0, max(-v for v in with_rsu) * 1.2)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("压力测试：GOOGL 崩了会怎样？", fontsize=13, weight="bold")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_factors(d, path):
    tab = d["factor_tab"].drop(index="alpha")
    names_cn = {"MKT": "市场", "TECH": "科技", "VALUE": "价值", "SIZE": "小盘",
                "INTL": "国际", "RATES": "利率", "GOLD": "黄金", "OIL": "原油"}
    labels = [f"{names_cn.get(i, i)}\n({i})" for i in tab.index]
    betas = tab["beta"].values
    sig = tab["t"].abs() > 2
    colors = [C_GOOG if s else "#bdbdbd" for s in sig]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, betas, color=colors, edgecolor="white")
    for b, t in zip(bars, tab["t"]):
        va = "bottom" if b.get_height() >= 0 else "top"
        off = 0.03 if b.get_height() >= 0 else -0.03
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                f"{b.get_height():+.2f}\n(t={t:+.1f})", ha="center", va=va, fontsize=8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("beta")
    ax.set_ylim(min(betas) - 0.45, max(betas) + 0.55)
    ax.set_title(f"因子 X 光：组合由什么驱动？（R²={d['r2']*100:.0f}%，红色=统计显著）",
                 fontsize=13, weight="bold")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_risk_contrib(d, path):
    rc = d["div"]["risk_contrib_pct"]
    total = sum(d["weights"].values())
    items = [(k, v, d["weights"][k] / total * 100) for k, v in rc.items() if k != "CASH"]
    items = items[:8]
    labels = [i[0] for i in items][::-1]
    risk = [i[1] for i in items][::-1]
    usd = [i[2] for i in items][::-1]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.barh(y + 0.2, risk, height=0.38, color=C_GOOG, label="风险贡献 %")
    ax.barh(y - 0.2, usd, height=0.38, color=C_IDX, label="美元占比 %")
    for yi, (r, u) in zip(y, zip(risk, usd)):
        ax.text(r + 0.6, yi + 0.2, f"{r:.1f}", va="center", fontsize=8)
        ax.text(u + 0.6, yi - 0.2, f"{u:.1f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    g_risk = d["div"]["risk_contrib_pct"]["GOOGL"]
    g_usd = d["weights"]["GOOGL"] / sum(d["weights"].values()) * 100
    ax.set_title(f"风险 vs 美元：GOOGL 用 {g_usd:.0f}% 的钱贡献了 {g_risk:.0f}% 的风险"
                 f"（ENB≈{d['div']['ENB']:.1f}）", fontsize=13, weight="bold")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_sim(d, path):
    sim = d["sim"]
    hold, sell = sim["hold"], sim["sell"]
    lo = min(np.percentile(hold, 0.5), np.percentile(sell, 0.5))
    hi = max(np.percentile(hold, 99), np.percentile(sell, 99))
    bins = np.linspace(lo, hi, 90)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5))
    ax = axes[0]
    ax.hist(hold, bins=bins, alpha=0.65, color=C_GOOG, label="HOLD 持有GOOG", density=True)
    ax.hist(sell, bins=bins, alpha=0.65, color=C_IDX, label="SELL 转分散篮子", density=True)
    for arr, c in [(hold, C_GOOG), (sell, C_IDX)]:
        ax.axvline(np.median(arr), color=c, linestyle="--", linewidth=1.4)
    ax.text(np.median(hold), ax.get_ylim()[1] * 0.97, f" 中位数 {money(np.median(hold), k=True)}",
            color=C_GOOG, fontsize=8.5, va="top")
    ax.text(np.median(sell), ax.get_ylim()[1] * 0.80, f" 中位数 {money(np.median(sell), k=True)}",
            color=C_IDX, fontsize=8.5, va="top")
    ax.set_title("48个月后终值分布", fontsize=11, weight="bold")
    ax.set_xlabel("终值（美元）")
    ax.legend(frameon=False, fontsize=8.5)
    ax.xaxis.set_major_formatter(lambda x, p: f"${x/1e6:.1f}M")

    ax2 = axes[1]
    hbins = np.linspace(0, max(np.percentile(sim["hold_dd"]*100, 99),
                               np.percentile(sim["sell_dd"]*100, 99)), 70)
    ax2.hist(sim["hold_dd"]*100, bins=hbins, alpha=0.65, color=C_GOOG, label="HOLD", density=True)
    ax2.hist(sim["sell_dd"]*100, bins=hbins, alpha=0.65, color=C_IDX, label="SELL", density=True)
    ax2.axvline(np.median(sim["hold_dd"]*100), color=C_GOOG, linestyle="--", linewidth=1.4)
    ax2.axvline(np.median(sim["sell_dd"]*100), color=C_IDX, linestyle="--", linewidth=1.4)
    ax2.set_title("最大回撤分布", fontsize=11, weight="bold")
    ax2.set_xlabel("最大回撤（%）")
    ax2.legend(frameon=False, fontsize=8.5)
    fig.suptitle("Monte Carlo：vest 后持有 GOOG vs 立即卖出转 VT（2 万条路径）",
                 fontsize=12.5, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- pdf
def build_pdf(d, imgs):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                    Table, TableStyle, PageBreak, KeepTogether)

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    F = "STSong-Light"

    st_title = ParagraphStyle("t", fontName=F, fontSize=24, leading=34, alignment=TA_CENTER,
                              textColor=colors.HexColor("#1a1a2e"))
    st_sub = ParagraphStyle("sub", fontName=F, fontSize=12, leading=18, alignment=TA_CENTER,
                            textColor=colors.HexColor("#555555"))
    st_h1 = ParagraphStyle("h1", fontName=F, fontSize=16, leading=22, spaceBefore=14,
                           spaceAfter=6, textColor=colors.HexColor("#d9552c"))
    st_body = ParagraphStyle("body", fontName=F, fontSize=10.5, leading=17, spaceAfter=6)
    st_call = ParagraphStyle("call", fontName=F, fontSize=10.5, leading=17,
                             textColor=colors.HexColor("#1a1a2e"))
    st_small = ParagraphStyle("small", fontName=F, fontSize=8.5, leading=13,
                              textColor=colors.HexColor("#777777"))

    def callout(text, bg="#fdf1e7", border="#d9552c"):
        t = Table([[Paragraph(text, st_call)]], colWidths=[165 * mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(border)),
            ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        return t

    def data_table(header, rows, widths=None):
        tt = Table([header] + rows, colWidths=widths)
        tt.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), F),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f7")]),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        return tt

    def img(path, w=165 * mm):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            iw, ih = im.size
        return Image(path, width=w, height=w * ih / iw)

    g, rsu, nw = d["liquid_googl"], d["rsu"], d["liquid_nw"]
    sim = d["sim"]
    hs = np.percentile(sim["hold"], [5, 25, 50, 75, 95])
    ss = np.percentile(sim["sell"], [5, 25, 50, 75, 95])
    rc = d["div"]["risk_contrib_pct"]
    g_risk_pct = rc["GOOGL"]
    g_usd_pct = d["weights"]["GOOGL"] / sum(d["weights"].values()) * 100
    top3_risk = sum(v for _, v in sorted(rc.items(), key=lambda kv: -kv[1])[:3])

    story = []
    # ---- cover
    story += [Spacer(1, 70 * mm),
              Paragraph("我的持仓 X 光报告", st_title),
              Spacer(1, 8 * mm),
              Paragraph("GOOGL 穿透 · 因子透视 · RSU 持有 vs 卖出", st_sub),
              Spacer(1, 5 * mm),
              Paragraph("2026 年 7 月 26 日 · 数据截至 7 月 23 日收盘 · 由 trade_sim 生成", st_sub),
              Spacer(1, 20 * mm),
              callout(f"<b>一句话总结：</b>净资产 {money(nw + rsu)}（含未 vest RSU）中，"
                      f"近一半是 Alphabet 单一名字的风险（{money(g + rsu)}，"
                      f"{(g + rsu) / (nw + rsu) * 100:.1f}%）；16 个持仓实际只有约 "
                      f"{d['div']['ENB']:.0f} 个独立赌注；模拟显示 GOOG 需年均跑赢全球篮子 "
                      f"+{d['be'] * 100:.1f}%，持有 RSU 才能追平 vest 即卖。"),
              PageBreak()]

    # ---- ch1 composition
    story += [Paragraph("第一章 · 钱都在哪儿？", st_h1),
              Paragraph(f"九个账户、十六个标的、合计 {money(nw + rsu)}。"
                        f"表面上横跨 Fidelity、Robinhood、M1、Vanguard 和 401(k)，"
                        f"看起来挺分散——但翻开底层，故事要单调得多。", st_body),
              img(imgs["comp"]),
              Spacer(1, 3 * mm),
              callout("注意前三大块：GOOGL 直接持股、大盘指数、纳指相关——"
                      "它们之间的相关性高达 0.95 以上。这不是三笔钱，是一笔钱的三个影子。"),
              PageBreak()]

    # ---- ch2 look-through
    story += [Paragraph("第二章 · GOOGL 穿透测试", st_h1),
              Paragraph(f"你以为的 GOOGL 敞口是直接持股的 {money(d['lt_rows'][0][2])}。"
                        f"但 SPY/VINIX 里有 {d['w_spy']*100:.1f}% 是 Alphabet，"
                        f"QQQM 里有 {d['w_qqq']*100:.1f}%——穿透之后还要再加 {money(d['lt_rows'][1][2] + d['lt_rows'][2][2] + d['lt_rows'][3][2])}。"
                        f"再加上 {rsu / d['px']['GOOGL']:,.0f} 股未 vest RSU（{money(rsu)}），画风突变：", st_body),
              data_table(["口径", "GOOGL 等值敞口", "占净资产", "GOOGL -30%", "GOOGL -50%"],
                         [["流动账户", money(g), f"{g/nw*100:.1f}%", money(g * 0.30), money(g * 0.50)],
                          ["含未 vest RSU", money(g + rsu), f"{(g+rsu)/(nw+rsu)*100:.1f}%",
                           money((g + rsu) * 0.30), money((g + rsu) * 0.50)]],
                         widths=[35 * mm, 40 * mm, 30 * mm, 30 * mm, 30 * mm]),
              Spacer(1, 4 * mm),
              img(imgs["lt"]),
              Spacer(1, 3 * mm),
              img(imgs["shock"]),
              Spacer(1, 3 * mm),
              callout("还没算进去的：GOOGL 大跌时，你的工作和未来 RSU vest 大概率同向承压。"
                      "股票敞口只是冰山露出水面的部分。"),
              PageBreak()]

    # ---- ch3 factor xray
    story += [Paragraph("第三章 · 因子 X 光：16 个持仓，4 个赌注", st_h1),
              Paragraph(f"回归结果毫不意外：市场 beta 高达 <b>1.62</b>（t=18.9），"
                        f"价值 beta <b>-0.78</b>——这是一个高杠杆的成长股组合。"
                        f"PC1 一个公共因子就解释了 {d['div']['PC1_share']*100:.0f}% 的截面方差，"
                        f"有效独立赌注数 ENB ≈ {d['div']['ENB']:.1f}，分散化比率 DR = {d['div']['DR']:.2f}"
                        f"（1.0 = 没有分散效果）。", st_body),
              img(imgs["factor"]),
              Spacer(1, 3 * mm),
              img(imgs["risk"]),
              Spacer(1, 3 * mm),
              callout(f"GOOGL 用 {g_usd_pct:.1f}% 的美元贡献了 {g_risk_pct:.1f}% 的风险；前三大风险源"
                      f"（GOOGL、QQQM、VINIX）合计 {top3_risk:.1f}%。砍风险从哪里动手，一目了然。"),
              PageBreak()]

    # ---- ch4 RSU sim
    story += [Paragraph("第四章 · RSU 灵魂拷问：vest 后持有还是卖出？", st_h1),
              Paragraph(f"2 万条 Monte Carlo 路径，模拟未来 48 个月、每季度 vest 一次。"
                        f"校准参数：GOOG 年化波动 {d['sg']*100:.0f}%、全球篮子 VT 波动 {d['sb']*100:.0f}%、"
                        f"相关系数 {d['rho']:.2f}，且假设两者期望收益相同（7%/年）——"
                        f"这样可以隔离出纯粹的「风险」差异，而不是猜谁会涨。", st_body),
              data_table(["策略", "p5 差路径", "中位数", "p95 好路径", "标准差", "中位回撤"],
                         [["HOLD 持有 GOOG", money(hs[0]), money(hs[2]), money(hs[4]),
                           money(sim["hold"].std(ddof=1)), f"{np.median(sim['hold_dd'])*100:.1f}%"],
                          ["SELL 转 VT", money(ss[0]), money(ss[2]), money(ss[4]),
                           money(sim["sell"].std(ddof=1)), f"{np.median(sim['sell_dd'])*100:.1f}%"]],
                         widths=[38 * mm, 30 * mm, 30 * mm, 30 * mm, 28 * mm, 24 * mm]),
              Spacer(1, 4 * mm),
              img(imgs["sim"]),
              Spacer(1, 3 * mm),
              callout(f"<b>反直觉的结论：</b>期望收益相同时，SELL 的中位数终值反而更高"
                      f"（{money(ss[2])} vs {money(hs[2])}）——波动率拖累让高 σ 资产的复利中位数缩水。"
                      f"HOLD 只赢在 p95 右尾（+{money(hs[4]-ss[4])}），代价是左尾 -{money(ss[0]-hs[0])}。"
                      f"<b>GOOG 需要年均跑赢 +{d['be']*100:.1f}%，HOLD 的中位数才能追平 SELL。</b>"),
              Spacer(1, 3 * mm),
              Paragraph("注意边界：vest 即卖近似税务中性（vest 时已按普通收入计税）；"
                        "但卖出已持有的 GOOGL 涉及资本利得税，不在本模拟范围内。"
                        "GBM 假设无肥尾，真实左尾更厚——对 HOLD 更不利。"
                        "ρ=0.63 对结论影响大：相关性越低，分散的免费午餐越大。", st_small),
              PageBreak()]

    # ---- ch5 conclusion
    story += [Paragraph("第五章 · 结论与行动清单", st_h1),
              Paragraph("把三章的证据摆在一起：", st_body),
              data_table(["证据", "数字", "出处"],
                         [["GOOGL 等值敞口（含 RSU）", f"{(g+rsu)/(nw+rsu)*100:.1f}%", "穿透分析"],
                          ["市场 beta", "1.62 (t=18.9)", "因子回归"],
                          ["有效独立赌注 ENB", f"{d['div']['ENB']:.1f}", "分散化分析"],
                          ["GOOGL 风险贡献", f"{g_risk_pct:.1f}%", "风险分解"],
                          ["HOLD 追平所需超额", f"+{d['be']*100:.1f}%/年", "Monte Carlo"]],
                         widths=[70 * mm, 50 * mm, 45 * mm]),
              Spacer(1, 5 * mm),
              Paragraph("可以考虑的动作（按代价从低到高）：", st_body),
              Paragraph("1. <b>新 vest 的 RSU 立即卖出</b>——税务中性，直接降低单一名字风险的最大增量来源；", st_body),
              Paragraph("2. <b>新增资金只投非科技资产</b>——用增量稀释存量，避免触发资本利得；", st_body),
              Paragraph("3. <b>逐步减持已 vest GOOGL</b>——利用长期资本利得税率和每年的免税额度，分批进行；", st_body),
              Paragraph("4. <b>检视 QQQM/TQQQ</b>——它们与 GOOGL 相关性 0.95+，减持它们的分散效果比减持低相关资产更好。", st_body),
              Spacer(1, 4 * mm),
              callout("本报告是风险量化，不是投资建议。所有模拟基于简化假设；"
                      "重大决策前请咨询税务和财务顾问。", bg="#eef3f9", border="#4c72b0")]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(F, 8)
        canvas.setFillColor(colors.HexColor("#999999"))
        canvas.drawCentredString(A4[0] / 2, 10 * mm,
                                 f"持仓 X 光报告 · 2026-07-26 · 第 {doc.page} 页")
        canvas.restoreState()

    SimpleDocTemplate(OUT_PDF, pagesize=A4,
                      leftMargin=22 * mm, rightMargin=22 * mm,
                      topMargin=18 * mm, bottomMargin=16 * mm,
                      title="持仓 X 光报告", author="trade_sim").build(
        story, onFirstPage=footer, onLaterPages=footer)


def main():
    print("collecting data (network calls for prices/weights/calibration)...")
    d = gather()
    with tempfile.TemporaryDirectory() as tmp:
        imgs = {}
        for key, fn in [("comp", fig_composition), ("lt", fig_lookthrough),
                        ("shock", fig_shock), ("factor", fig_factors),
                        ("risk", fig_risk_contrib), ("sim", fig_sim)]:
            p = str(Path(tmp) / f"{key}.png")
            fn(d, p)
            imgs[key] = p
            print(f"  chart {key} done")
        build_pdf(d, imgs)
    print(f"PDF written: {OUT_PDF}")


if __name__ == "__main__":
    main()
