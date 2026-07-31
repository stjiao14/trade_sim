"""Generate an interactive single-file HTML report (portfolio_report.html).

Formal investment-report style with full mathematical derivations (MathJax,
inlined for offline use) and interactive Plotly charts (also inlined).

Usage: .venv/bin/python make_report_html.py
"""
import json

import numpy as np
import pandas as pd

import concentration_analysis as ca
import factor_xray as fx
import vest_diversify_sim as vs

OUT_HTML = "portfolio_report.html"

COLORS = dict(goog="#b5442a", rsu="#d98e4a", idx="#3f5f8f", idx2="#5b84b8",
              idx3="#8ba8cc", tech="#6b6391", tech2="#a89fc4", gold="#a8903a",
              cash="#8a8a8a", green="#2f7d57", grey="#bdbdbd")


def gather():
    d = {}
    df = pd.read_csv("holdings.csv")
    px = ca.fetch_prices()
    H, _, _ = ca.load_holdings_from_csv("holdings.csv", px)
    w_spy, _ = ca.fetch_alphabet_weight("SPY", 0.038)
    w_qqq, _ = ca.fetch_alphabet_weight("QQQ", 0.050)
    rows, liquid_googl, liquid_nw, rsu = ca.compute_lookthrough(H, px, w_spy, w_qqq)
    d.update(lt_rows=rows, liquid_googl=liquid_googl, liquid_nw=liquid_nw, rsu=rsu,
             w_spy=w_spy, w_qqq=w_qqq, holdings=df)

    weights, _ = fx.load_full_portfolio("holdings.csv")
    R, w, port = fx.portfolio_returns(weights)
    tab, r2, _ = fx.factor_xray(port)
    div = fx.diversification(R, w)
    d.update(factor_tab=tab, r2=r2, div=div, weights=weights)

    cfg, basket, n_paths, _ = vs._load_config()
    sg, sb, rho = vs._calibrate_yfinance(basket, 1260)
    sim = vs.simulate(cfg, 0.07, 0.07, sg, sb, rho, N=n_paths)
    be = vs.breakeven_drift(cfg, 0.07, sg, sb, rho)
    d.update(sim=sim, sg=sg, sb=sb, rho=rho, be=be, cfg=cfg)

    sweep_x, sweep_y = [], []
    for e in np.arange(0.0, 0.085, 0.005):
        r = vs.simulate(cfg, 0.07 + e, 0.07, sg, sb, rho, N=n_paths)
        sweep_x.append(round(float(e) * 100, 1))
        sweep_y.append(float(np.median(r["hold"]) - np.median(r["sell"])))
    d["sweep"] = (sweep_x, sweep_y)
    return d


def build_data(d):
    df = d["holdings"]
    v = df.set_index("ticker_symbol")["institution_value"].groupby(level=0).sum()
    rsu_val = df.loc[df.subtype == "rsu", "institution_value"].sum()
    sleeves = [
        ("GOOGL 直接持股", v.get("GOOGL", 0) - rsu_val, COLORS["goog"]),
        ("未归属 RSU", rsu_val, COLORS["rsu"]),
        ("大盘指数 (VINIX+SPY)", v.get("VINIX", 0) + v.get("SPY", 0), COLORS["idx"]),
        ("纳指相关 (QQQM+TQQQ)", v.get("QQQM", 0) + v.get("TQQQ", 0), COLORS["idx2"]),
        ("扩展市场 (VIEIX)", v.get("VIEIX", 0), COLORS["idx3"]),
        ("其它科技个股", sum(v.get(t, 0) for t in ["AMD", "MSFT", "AVGO", "TSM", "INTC", "BNTX", "ZM"]), COLORS["tech"]),
        ("主题ETF (DRAM)", v.get("DRAM", 0), COLORS["tech2"]),
        ("黄金 (GLD)", v.get("GLD", 0), COLORS["gold"]),
        ("现金类", df.loc[df.subtype == "cash", "institution_value"].sum(), COLORS["cash"]),
    ]
    sleeves = [s for s in sleeves if s[1] > 0]
    total = sum(s[1] for s in sleeves)

    rows = d["lt_rows"]
    g, rsu, nw = d["liquid_googl"], d["rsu"], d["liquid_nw"]
    rc = d["div"]["risk_contrib_pct"]
    wt = d["weights"]
    wtotal = sum(wt.values())
    risk_items = [(k, rc[k], wt[k] / wtotal * 100) for k in rc if k != "CASH"][:8]

    tab = d["factor_tab"].drop(index="alpha")
    names_cn = {"MKT": "市场", "TECH": "科技", "VALUE": "价值", "SIZE": "小盘",
                "INTL": "国际", "RATES": "利率", "GOLD": "黄金", "OIL": "原油"}

    sim = d["sim"]
    hold, sell = sim["hold"], sim["sell"]
    lo = min(np.percentile(hold, 0.5), np.percentile(sell, 0.5))
    hi = max(np.percentile(hold, 99), np.percentile(sell, 99))
    edges = np.linspace(lo, hi, 61)
    ch, _ = np.histogram(hold, bins=edges)
    cs, _ = np.histogram(sell, bins=edges)
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()

    dd_edges = np.linspace(0, max(np.percentile(sim["hold_dd"], 99),
                                  np.percentile(sim["sell_dd"], 99)) * 100, 51)
    dh, _ = np.histogram(sim["hold_dd"] * 100, bins=dd_edges)
    ds, np_ = np.histogram(sim["sell_dd"] * 100, bins=dd_edges)
    dd_centers = ((dd_edges[:-1] + dd_edges[1:]) / 2).tolist()

    hs = np.percentile(hold, [5, 25, 50, 75, 95])
    ss = np.percentile(sell, [5, 25, 50, 75, 95])
    sx, sy = d["sweep"]

    return dict(
        total=float(d["liquid_nw"] + d["rsu"]),
        nw=float(nw), rsu=float(rsu), g=float(g),
        g_share_liquid=float(g / nw * 100),
        g_share_total=float((g + rsu) / (nw + rsu) * 100),
        w_spy=float(d["w_spy"] * 100), w_qqq=float(d["w_qqq"] * 100),
        lt_extra=float(rows[1][2] + rows[2][2] + rows[3][2]),
        lt_direct=float(rows[0][2]),
        lt_mv=dict(direct=float(rows[0][1]), spy=float(rows[1][1]),
                   qqq=float(rows[2][1]), tqqq=float(rows[3][1])),
        lookthrough=dict(
            names=["直接持股", "SPY 穿透", "QQQ 穿透", "TQQQ 3x穿透", "未归属 RSU"],
            values=[float(rows[0][2]), float(rows[1][2]), float(rows[2][2]),
                    float(rows[3][2]), float(rsu)],
            colors=[COLORS["goog"], COLORS["idx"], COLORS["idx2"], COLORS["idx3"], COLORS["rsu"]]),
        shock=dict(liquid=[float(g * .30), float(g * .50)],
                   with_rsu=[float((g + rsu) * .30), float((g + rsu) * .50)]),
        sleeves=dict(labels=[s[0] for s in sleeves], values=[float(s[1]) for s in sleeves],
                     colors=[s[2] for s in sleeves], total=float(total)),
        factors=dict(labels=[f"{names_cn.get(i, i)} ({i})" for i in tab.index],
                     beta=[float(x) for x in tab["beta"]],
                     t=[float(x) for x in tab["t"]]),
        r2=float(d["r2"] * 100),
        div=dict(ENB=float(d["div"]["ENB"]), DR=float(d["div"]["DR"]),
                 PC1=float(d["div"]["PC1_share"] * 100)),
        risk=dict(labels=[i[0] for i in risk_items],
                  risk=[float(i[1]) for i in risk_items],
                  usd=[float(i[2]) for i in risk_items]),
        g_risk=float(rc["GOOGL"]),
        g_usd=float(wt["GOOGL"] / wtotal * 100),
        top3_risk=float(sum(vv for _, vv in sorted(rc.items(), key=lambda kv: -kv[1])[:3])),
        sim=dict(centers=[round(c) for c in centers],
                 hold=ch.tolist(), sell=cs.tolist(),
                 dd_centers=[round(c, 1) for c in dd_centers],
                 dd_hold=dh.tolist(), dd_sell=ds.tolist(),
                 hold_med=float(np.median(hold)), sell_med=float(np.median(sell)),
                 hold_p5=float(hs[0]), sell_p5=float(ss[0]),
                 hold_p95=float(hs[4]), sell_p95=float(ss[4]),
                 hold_std=float(hold.std(ddof=1)), sell_std=float(sell.std(ddof=1)),
                 hold_dd_med=float(np.median(sim["hold_dd"]) * 100),
                 sell_dd_med=float(np.median(sim["sell_dd"]) * 100)),
        sweep=dict(x=sx, y=[round(y0) for y0 in sy], be=float(d["be"] * 100)),
        calib=dict(sg=float(d["sg"] * 100), sb=float(d["sb"] * 100), rho=float(d["rho"])),
        drag=float((d["sg"] ** 2 - d["sb"] ** 2) / 2 * 100),
    )


GLOSSARY = [
    ("RSU", "限制性股票单位（Restricted Stock Unit）。公司授予的股票，按既定时间表分批归属。归属时按当日市价折算为普通收入计税，因此归属即卖出近似税务中性。"),
    ("vest（归属）", "RSU 分批转为可自由处置股票的过程。未归属部分在离职时通常会作废，其价值同时受股价与在职状态影响。"),
    ("穿透敞口（look-through exposure）", "投资组合对某一底层资产的直接与间接暴露之和。指数基金按其成分权重折算间接暴露，例如 SPY 中 Alphabet 权重约 5.8%，则每持有 100 美元 SPY 相当于间接持有 5.8 美元 Alphabet。"),
    ("beta", "资产收益率对某一因子收益率的回归斜率，衡量系统性暴露。beta=1.62 表示因子每变动 1%，组合平均同向变动 1.62%。"),
    ("t 值（t-statistic）", "回归系数与其标准误之比，用于检验系数是否显著异于零。通常以 |t|>2 作为 5% 显著性水平的近似判据。"),
    ("R²（决定系数）", "回归模型解释的方差占样本总方差的比例。R²=82% 表示组合约八成的日度波动可由所选因子解释。"),
    ("相关系数 ρ", "两项资产收益率线性相关程度，取值 [-1, 1]。0.95 以上表明二者走势几乎一致；本报告中 GOOG 与全球股票篮子的相关系数为 0.63。"),
    ("波动率 σ", "收益率的年化标准差。GOOG σ≈32% 表示在正态假设下，约 68% 的年份其收益落在均值 ±32% 区间内。"),
    ("ENB（有效独立赌注数）", "Effective Number of Bets，基于持仓收益率相关系数矩阵特征值计算的集中度指标，数学形式为逆参与率。取值 1 表示完全集中，N 表示 N 个完全独立且等权的风险来源。"),
    ("PC1（第一主成分）", "持仓收益率协方差结构中方差最大的正交方向，可理解为组合中最主要的共同驱动因素。PC1 占比越高，组合越由单一宏观因素主导。"),
    ("DR（分散化比率）", "各资产波动率的加权平均与组合实际波动率之比。下限为 1.0（完全相关、无分散收益），数值越大表明组合内部的波动抵消越充分。"),
    ("Monte Carlo 模拟", "通过随机抽样生成大量可能的价格路径，统计目标变量的经验分布（分位数、回撤等），用于在解析解不可得时评估策略的风险收益特征。"),
    ("GBM（几何布朗运动）", "连续时间股价模型：对数收益服从独立正态增量，价格过程为 dS/S = μdt + σdW。该模型不含肥尾、跳跃与波动率聚集，会低估极端事件的频率。"),
    ("分位数 p5 / p50 / p95", "经验分布中累计概率分别为 5%、50%、95% 的取值。p5 与 p95 分别刻画不利与有利情景的边界，p50（中位数）刻画典型结果。"),
    ("最大回撤", "净值自历史最高点到其后最低点的最大跌幅，衡量持有期间需要承受的最深账面亏损。"),
    ("波动率拖累（volatility drag）", "几何平均收益率 ≈ 算术平均收益率 − σ²/2。在期望收益相同的前提下，波动率越高，复利增长的中位数越低。"),
    ("打平超额收益（breakeven drift）", "使两种策略终值中位数相等所需的标的资产年化超额收益。本报告中指使「持有 RSU」中位数追平「归属即卖出」所需的 GOOG 相对全球股票篮子的年化超额。"),
    ("VT", "Vanguard Total World Stock ETF，覆盖全球约一万只股票的市值加权指数，本报告以其作为完全分散化权益篮子的代理。"),
    ("资本利得税", "对资产卖出价格与计税成本之间差额征收的税。RSU 归属即卖出因计税成本等于归属日市价而近似无资本利得；卖出长期持有的已增值股票则会触发，构成减持的主要摩擦成本。"),
]


def _js(path):
    with open(path, encoding="utf-8") as f:
        return f.read().replace("</script", "<\\/script")


def build_html(data):
    dj = json.dumps(data, ensure_ascii=False)
    glossary_cards = "\n".join(
        f'<div class="gcard"><div class="gterm">{t}</div><div class="gdef">{e}</div></div>'
        for t, e in GLOSSARY)
    terms_js = json.dumps(dict(GLOSSARY), ensure_ascii=False)

    html = HTML_TEMPLATE
    subs = {
        "%%PLOTLY%%": _js("plotly.min.js"),
        "%%MATHJAX%%": _js("tex-svg.js"),
        "%%DATA%%": dj,
        "%%GLOSSARY_CARDS%%": glossary_cards,
        "%%TERMS%%": terms_js,
        "%%F_DIRECT%%": f"{data['lt_mv']['direct']:,.0f}",
        "%%F_SPYV%%": f"{data['lt_mv']['spy']:,.0f}",
        "%%F_QQQV%%": f"{data['lt_mv']['qqq']:,.0f}",
        "%%F_TQQQV%%": f"{data['lt_mv']['tqqq']:,.0f}",
        "%%F_RSU%%": f"{data['rsu']:,.0f}",
        "%%F_ETOTAL%%": f"{data['g'] + data['rsu']:,.0f}",
        "%%F_NW%%": f"{data['total']:,.0f}",
        "%%F_SPYW%%": f"{data['w_spy'] / 100:.3f}",
        "%%F_QQQW%%": f"{data['w_qqq'] / 100:.3f}",
        "%%F_SHARE%%": f"{data['g_share_total']:.1f}",
        "%%F_SG%%": f"{data['calib']['sg']:.1f}",
        "%%F_SB%%": f"{data['calib']['sb']:.1f}",
        "%%F_RHO%%": f"{data['calib']['rho']:.2f}",
        "%%F_DRAG%%": f"{data['drag']:.1f}",
        "%%F_BE%%": f"{data['sweep']['be']:.1f}",
    }
    for k, v in subs.items():
        html = html.replace(k, v)
    return html


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>个人投资组合风险分析报告 · 2026-07-26</title>
<script>
MathJax = { tex: { inlineMath: [["\\(", "\\)"]], packages: ["base", "ams", "noundefined"] },
            svg: { fontCache: "global" } };
</script>
<script>%%MATHJAX%%</script>
<script>%%PLOTLY%%</script>
<style>
:root {
  --goog: #b5442a; --idx: #3f5f8f; --ink: #1f2430;
  --muted: #5d6673; --bg: #f7f6f3; --card: #ffffff; --line: #e5e2db;
  --accent-bg: #f9f1ec; --blue-bg: #eef2f7;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--ink); line-height: 1.85; font-size: 15.5px;
}
.wrap { max-width: 920px; margin: 0 auto; padding: 0 20px 80px; }

.hero {
  background: #1f2430; color: #fff; padding: 64px 20px 48px; text-align: center;
  border-bottom: 4px solid var(--goog);
}
.hero h1 { font-size: 32px; letter-spacing: 4px; margin-bottom: 10px; font-weight: 600; }
.hero .sub { color: #9aa3b2; font-size: 14px; margin-bottom: 34px; letter-spacing: 1px; }
.stats { display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; }
.stat {
  border: 1px solid rgba(255,255,255,.18); border-radius: 6px;
  padding: 14px 24px; min-width: 170px; text-align: center;
}
.stat .num { font-size: 24px; font-weight: 600; color: #e8a37c; }
.stat .lbl { font-size: 12px; color: #9aa3b2; margin-top: 4px; letter-spacing: .5px; }

nav {
  position: sticky; top: 0; z-index: 50; background: rgba(247,246,243,.95);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--line);
  padding: 10px 0; text-align: center;
}
nav a { color: var(--muted); text-decoration: none; font-size: 13px; margin: 0 12px; }
nav a:hover { color: var(--goog); }

section { margin-top: 52px; }
h2 {
  font-size: 22px; margin-bottom: 4px; font-weight: 600;
  border-bottom: 2px solid var(--goog); padding-bottom: 8px; display: inline-block;
}
h3 { font-size: 17px; margin: 26px 0 8px; font-weight: 600; color: var(--ink); }
.lead { color: var(--muted); margin: 10px 0 16px; }
.card {
  background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 22px 26px; margin: 16px 0; box-shadow: 0 1px 3px rgba(31,36,48,.04);
}
.callout {
  background: var(--accent-bg); border-left: 3px solid var(--goog); border-radius: 4px;
  padding: 15px 20px; margin: 18px 0;
}
.callout.blue { background: var(--blue-bg); border-left-color: var(--idx); }
.chart { width: 100%; height: 400px; }
.chart.tall { height: 440px; }
.fallback { color: var(--muted); font-size: 14px; padding: 40px; text-align: center; }

.formula {
  background: #fbfaf7; border: 1px solid var(--line); border-radius: 6px;
  padding: 14px 20px; margin: 14px 0; overflow-x: auto; font-size: 14.5px;
}
.formula .flabel { color: var(--muted); font-size: 12.5px; letter-spacing: 1px; margin-bottom: 4px; }
p.eq { margin: 10px 0; }
.where { color: var(--muted); font-size: 13.5px; margin-top: 6px; }

.term { color: var(--idx); border-bottom: 1px dashed var(--idx); cursor: help; }
#popover {
  position: fixed; z-index: 99; display: none; max-width: 340px;
  background: var(--ink); color: #fff; font-size: 13.5px; line-height: 1.6;
  border-radius: 8px; padding: 14px 16px; box-shadow: 0 8px 30px rgba(0,0,0,.25);
}
#popover .pt { color: #e8a37c; font-weight: 600; margin-bottom: 4px; }

table.dt { width: 100%; border-collapse: collapse; font-size: 14.5px; }
table.dt th {
  background: var(--ink); color: #fff; padding: 10px 14px; text-align: left; font-weight: 600;
}
table.dt th:not(:first-child), table.dt td:not(:first-child) { text-align: right; }
table.dt td { padding: 10px 14px; border-bottom: 1px solid var(--line); }
table.dt tr:nth-child(even) td { background: #faf8f4; }

.slidebox { text-align: center; padding: 10px 12px 2px; border-top: 1px solid var(--line); }
.slidebox input[type=range] { width: 70%; accent-color: var(--goog); }
.verdict { font-size: 15.5px; font-weight: 600; margin-top: 8px; }
.verdict .good { color: var(--goog); }
.verdict .bad { color: var(--idx); }

.ggrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 14px; }
.gcard { background: var(--card); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
.gterm { font-weight: 600; color: var(--goog); margin-bottom: 4px; font-size: 14px; }
.gdef { font-size: 13px; color: #4b5563; line-height: 1.65; }

footer { text-align: center; color: var(--muted); font-size: 12.5px; margin-top: 64px; line-height: 2; }
ol.actions { padding-left: 22px; }
ol.actions li { margin: 10px 0; }
b.hl { color: var(--goog); }
</style>
</head>
<body>

<div class="hero">
  <h1>个人投资组合风险分析报告</h1>
  <div class="sub">集中度风险 · 因子暴露 · RSU 处置策略 · 数据截至 2026-07-23 收盘</div>
  <div class="stats" id="heroStats"></div>
</div>

<nav>
  <a href="#ch1">资产配置</a><a href="#ch2">集中度分析</a><a href="#ch3">因子暴露</a>
  <a href="#ch4">RSU 策略模拟</a><a href="#ch5">结论</a><a href="#glossary">附录</a>
</nav>

<div class="wrap">

<section id="ch0">
  <h2>摘要</h2>
  <div class="card">
    <p>本报告基于九个账户、十六个证券标的的完整持仓数据（合计 <b id="nwTotal"></b>，含未归属 RSU），
    从三个维度量化组合风险：（1）对 Alphabet 单一标的的穿透敞口；（2）系统性因子暴露与真实分散化程度；
    （3）RSU 归属后「继续持有」与「归属即卖出并转入分散篮子」两种策略的风险收益分布。
    全部结论均给出计算过程与统计依据，蒙特卡洛部分使用 20,000 条模拟路径。</p>
  </div>
</section>

<section id="ch1">
  <h2>1 &nbsp;资产配置结构</h2>
  <p class="lead">组合净资产定义为各持仓市值与未归属 RSU 折算市值之和。悬停图表可查看各组成部分的金额与占比。</p>
  <div class="formula">
    <div class="flabel">式 (1) · 净资产</div>
    \[ NW \;=\; \sum_{i} V_i \;+\; V_{RSU} \;=\; %%F_NW%% \]
    <div class="where">其中 \(V_i\) 为第 \(i\) 项持仓市值，\(V_{RSU}\) 为未归属 RSU 按当前市价折算的经济敞口。</div>
  </div>
  <div class="card"><div id="figComp" class="chart tall"></div></div>
  <div class="callout">结构观察：前三大组成部分（GOOGL 直接持股、大盘指数基金、纳指相关产品）的日收益率两两<span class="term">相关系数 ρ</span>均在 0.95 以上，名义上的多元配置在经济实质上高度同质。</div>
</section>

<section id="ch2">
  <h2>2 &nbsp;单一标的集中度：Alphabet 穿透敞口</h2>
  <h3>2.1 穿透计算框架</h3>
  <p>组合对 Alphabet 的真实敞口由四部分构成：直接持股、指数基金穿透（SPY 类与 QQQ 类）、杠杆 ETF 的名义敞口（TQQQ 按 3 倍计）、以及未归属 RSU。ETF 内 Alphabet（GOOGL+GOOG）权重取自基金公司披露数据，当前分别为 SPY \(w_{SPY}=%%F_SPYW%%\)、QQQ \(w_{QQQ}=%%F_QQQW%%\)。</p>
  <div class="formula">
    <div class="flabel">式 (2) · GOOGL 等值穿透敞口</div>
    \[ E_{GOOGL} \;=\; V_{dir} \;+\; w_{SPY}\,V_{SPY类} \;+\; w_{QQQ}\left(V_{QQQ类} + 3\,V_{TQQQ}\right) \;+\; V_{RSU} \]
  </div>
  <div class="formula">
    <div class="flabel">式 (2′) · 代入当前持仓</div>
    \[ E_{GOOGL} = %%F_DIRECT%% + %%F_SPYW%% \times %%F_SPYV%% + %%F_QQQW%% \times (%%F_QQQV%% + 3 \times %%F_TQQQV%%) + %%F_RSU%% \approx %%F_ETOTAL%% \]
  </div>
  <div class="card"><table class="dt" id="ltTable"></table></div>
  <div class="card"><div id="figLt" class="chart" style="height:280px"></div></div>

  <h3>2.2 压力测试</h3>
  <p>在 Alphabet 价格冲击 \(\delta\) 下，组合账面损失按敞口线性放大：</p>
  <div class="formula">
    <div class="flabel">式 (3) · 冲击损失</div>
    \[ L(\delta) \;=\; -\,E_{GOOGL} \cdot \delta, \qquad \delta \in \{30\%,\; 50\%\} \]
    <div class="where">该式仅计 Alphabet 直接冲击；考虑科技板块高相关性（见第 3 节），实际组合损失将大于此估计。</div>
  </div>
  <div class="card"><div id="figShock" class="chart" style="height:340px"></div></div>
  <div class="callout">补充说明：上述测算未包含人力资本渠道——Alphabet 深度回撤通常伴随薪酬中股票部分缩水与归属价值下降，单一标的风险的实际暴露高于证券账户口径。</div>
</section>

<section id="ch3">
  <h2>3 &nbsp;因子暴露与真实分散化程度</h2>
  <h3>3.1 因子回归模型</h3>
  <p>以组合日收益率 \(r_{p,t} = \sum_i w_i r_{i,t}\) 为因变量，对八个 ETF 因子代理（市场 SPY、科技 XLK、价值 IWD、小盘 IWM、国际 EFA、利率 IEF、黄金 GLD、原油 USO）作普通最小二乘回归：</p>
  <div class="formula">
    <div class="flabel">式 (4) · 因子模型与 OLS 估计</div>
    \[ r_{p,t} = \alpha + \sum_{k} \beta_k f_{k,t} + \varepsilon_t, \qquad
       \hat{\beta} = (X^\top X)^{-1} X^\top r_p, \qquad
       t_k = \frac{\hat{\beta}_k}{\hat{\sigma}\,\sqrt{(X^\top X)^{-1}_{kk}}} \]
    <div class="where">样本为过去 750 个自然日（约 3 年）的日收益。因子代理间存在较高共线性（条件数约 550），单个 \(\beta_k\) 的点估计不宜按正交暴露解读，应以 \(R^2\) 与主导暴露为准。</div>
  </div>
  <div class="card"><div id="figFactor" class="chart"></div></div>

  <h3>3.2 有效赌注数与分散化指标</h3>
  <p>设持仓收益率相关系数矩阵 \(C\) 的特征值为 \(\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_N\)，组合权重向量为 \(w\)，协方差矩阵为 \(\Sigma\)：</p>
  <div class="formula">
    <div class="flabel">式 (5) · 三个分散化指标</div>
    \[ \mathrm{ENB} = \frac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}, \qquad
       \mathrm{PC1} = \frac{\lambda_1}{\sum_i \lambda_i}, \qquad
       \mathrm{DR} = \frac{\sum_i w_i \sigma_i}{\sqrt{w^\top \Sigma w}} \]
    <div class="where">ENB 为逆参与率形式：完全集中时取 1，N 个独立等权赌注时取 N。DR 的下限为 1（完全相关、无分散收益）。</div>
  </div>
  <div class="card"><div id="figRisk" class="chart tall"></div></div>

  <h3>3.3 风险贡献分解</h3>
  <p>利用 Euler 分解，组合方差可完全分摊到各持仓：</p>
  <div class="formula">
    <div class="flabel">式 (6) · 风险贡献</div>
    \[ RC_i = \frac{w_i\,(\Sigma w)_i}{w^\top \Sigma w}, \qquad \sum_i RC_i = 1 \]
    <div class="where">\(RC_i\) 同时反映持仓权重、自身波动率及与其余持仓的相关性，是识别「以较小资金贡献较大风险」标的的规范工具。</div>
  </div>
  <div class="callout" id="riskCallout"></div>
</section>

<section id="ch4">
  <h2>4 &nbsp;RSU 处置策略的蒙特卡洛比较</h2>
  <h3>4.1 模型设定与参数校准</h3>
  <p>设 GOOG 与分散篮子（以 <span class="term">VT</span> 代理）价格均服从<span class="term">GBM</span>，两者通过对数收益的相关系数 \(\rho\) 联动。离散化为月度步长 \(\Delta t = 1/12\)：</p>
  <div class="formula">
    <div class="flabel">式 (7) · 相关几何布朗运动</div>
    \[ \ln \frac{S_{t+\Delta t}}{S_t} = \left(\mu - \tfrac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t}\, Z_t,
       \qquad \begin{pmatrix} Z_t^{g} \\ Z_t^{b} \end{pmatrix} = L\,\varepsilon_t, \quad
       L L^\top = \begin{pmatrix} \sigma_g^2 & \rho\,\sigma_g\sigma_b \\ \rho\,\sigma_g\sigma_b & \sigma_b^2 \end{pmatrix} \]
    <div class="where">\(L\) 为协方差矩阵的 Cholesky 因子，\(\varepsilon_t\) 为独立标准正态向量。</div>
  </div>
  <div class="formula">
    <div class="flabel">式 (8) · 参数校准（过去 1260 个交易日，约 5 年）</div>
    \[ \hat{\sigma} = \operatorname{std}(r_t)\sqrt{252}, \qquad \hat{\rho} = \operatorname{corr}(r_t^{g}, r_t^{b}),
       \qquad r_t = \ln \frac{P_t}{P_{t-1}} \]
    <div class="where">校准结果：\(\hat{\sigma}_g = %%F_SG%%\%\)、\(\hat{\sigma}_b = %%F_SB%%\%\)、\(\hat{\rho} = %%F_RHO%%\)。基准假设 \(\mu_g = \mu_b = 7\%\)，以隔离纯风险维度差异。</div>
  </div>

  <h3>4.2 两种策略的终值构造</h3>
  <p>记 \(G_t, B_t\) 为归一化价格路径（\(G_0 = B_0 = 1\)），未归属 RSU 共 \(N\) 股、分 \(J\) 次于月末 \(\{t_j\}\) 等额归属，归属参考价 \(P_0\)，流动账户中 GOOG 市值 \(L_g\)、其余资产市值 \(L_b\)：</p>
  <div class="formula">
    <div class="flabel">式 (9) · HOLD（归属后继续持有 GOOG）</div>
    \[ V_H(T) = (L_g + N P_0)\,G_T + L_b\,B_T \]
  </div>
  <div class="formula">
    <div class="flabel">式 (10) · SELL（每次归属即卖出并转入篮子）</div>
    \[ V_S(T) = L_g\,G_T + L_b\,B_T + \sum_{j=1}^{J} \underbrace{\frac{N}{J} P_0\, G_{t_j}}_{\text{第 }j\text{ 次归属卖出所得}} \cdot \underbrace{\frac{B_T}{B_{t_j}}}_{\text{转入篮子后增值}} + \; N_u(T)\,P_0\,G_T \]
    <div class="where">\(N_u(T)\) 为期末仍未归属股数。归属环节税负两策略相同（归属即按普通收入计税），故归属即卖出近似税务中性。</div>
  </div>
  <div class="card"><table class="dt" id="simTable"></table></div>
  <div class="card"><div id="figSim" class="chart tall"></div></div>

  <h3>4.3 波动率拖累：中位数为何偏向 SELL</h3>
  <p>对数正态分布的中位数为 \(\exp\left((\mu - \sigma^2/2)\,T\right)\)，即复利增长的典型路径以 \(\sigma^2/2\) 的速率折损。两种策略在该项上的年化差异近似为：</p>
  <div class="formula">
    <div class="flabel">式 (11) · 波动率拖累差异</div>
    \[ \frac{\sigma_g^2 - \sigma_b^2}{2} = \frac{0.318^2 - 0.162^2}{2} \approx %%F_DRAG%%\% \;/\;\text{年} \]
    <div class="where">这解释了表观上「期望收益相同、中位数却不同」的现象：差异并非来自方向判断，而是方差对复利路径的系统性侵蚀。</div>
  </div>

  <h3>4.4 打平超额收益与情景分析</h3>
  <p>定义打平超额 \(e^*\) 为使两策略终值中位数相等的 GOOG 年化超额收益，以二分法在 \([0, 10\%]\) 上求解（容差 0.2%）：</p>
  <div class="formula">
    <div class="flabel">式 (12) · 打平条件</div>
    \[ e^* : \quad \operatorname{median}\left[ V_H\!\left(\mu_g = \mu + e^*\right) \right] = \operatorname{median}\left[ V_S(\mu_g = \mu) \right]
       \;\;\Longrightarrow\;\; e^* \approx %%F_BE%%\% \;/\; \text{年} \]
    <div class="where">\(e^*\) 高于式 (11) 的静态拖累估计，原因是 SELL 的分散化收益随归属进度在 48 个月内逐步兑现，削弱了拖累优势的累积。</div>
  </div>
  <div class="card">
    <div id="figSweep" class="chart" style="height:360px"></div>
    <div class="slidebox">
      <label>情景假设：GOOG 相对篮子的年均超额收益 <b id="driftLbl"></b></label><br>
      <input type="range" id="drift" min="0" max="8" step="0.5" value="4.5">
      <div class="verdict" id="verdict"></div>
    </div>
  </div>
  <div class="callout" id="simCallout"></div>
</section>

<section id="ch5">
  <h2>5 &nbsp;结论与风险提示</h2>
  <div class="card"><table class="dt" id="evTable"></table></div>
  <div class="card">
    <p style="margin-bottom:10px">基于上述证据，可供考虑的组合调整路径（按实施摩擦由低到高排序）：</p>
    <ol class="actions">
      <li><b class="hl">新归属 RSU 即刻卖出</b>：税务近似中性，可直接阻断单一标的敞口的最大增量来源；</li>
      <li><b class="hl">新增资金仅配置低相关性资产</b>：以增量稀释存量集中度，不触发任何税负；</li>
      <li><b class="hl">分年度减持已归属 GOOGL</b>：利用长期<span class="term">资本利得税</span>率与年度免税空间，控制单次税基；</li>
      <li><b class="hl">评估 QQQM / TQQQ 的必要性</b>：二者与 GOOGL 相关性逾 0.95，减持其对降低组合方差的边际效果优于减持低相关资产。</li>
    </ol>
  </div>
  <div class="callout blue">
    <b>模型与数据局限：</b>（1）<span class="term">GBM</span> 假设不含肥尾与波动率聚集，极端左尾风险被低估，且对 HOLD 策略的影响方向更为不利；
    （2）结论对 <span class="term">相关系数 ρ</span> 与篮子代理的选择敏感，ρ 越低分散化收益越大；
    （3）因子代理间共线性较高，单个 beta 的点估计不宜按正交因子解读；
    （4）卖出已归属 GOOGL 涉及<span class="term">资本利得税</span>，未纳入模拟；
    （5）本报告为风险量化研究，不构成投资建议，重大决策请咨询税务与财务顾问。
  </div>
</section>

<section id="glossary">
  <h2>附录 &nbsp;名词表</h2>
  <p class="lead">正文中带虚线下划线的术语均可点击查看释义，下表为完整版本。</p>
  <div class="ggrid">%%GLOSSARY_CARDS%%</div>
</section>

<footer>
  个人投资组合风险分析报告 · 由 trade_sim 生成 · 2026-07-26<br>
  数据来源：Yahoo Finance（价格与基金持仓权重）· 模拟：20,000 条蒙特卡洛路径 · 仅供个人研究，不构成投资建议
</footer>
</div>

<div id="popover"><div class="pt"></div><div class="pd"></div></div>

<script>
const D = %%DATA%%;
const TERMS = %%TERMS%%;
const fmt$ = x => "$" + Math.round(x).toLocaleString("en-US");
const fmt$k = x => "$" + Math.round(x / 1000).toLocaleString("en-US") + "k";
const C = {goog:"#b5442a", rsu:"#d98e4a", idx:"#3f5f8f", grey:"#bdbdbd", ink:"#1f2430"};
const LAY = {font:{family:"-apple-system, PingFang SC, Hiragino Sans GB, sans-serif", color:"#1f2430"},
             margin:{t:50, r:30, b:50, l:60}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)"};
const CFG = {responsive:true, displayModeBar:false};

/* hero */
document.getElementById("nwTotal").textContent = fmt$(D.total);
document.getElementById("heroStats").innerHTML = [
  [fmt$(D.total), "总净资产（含未归属 RSU）"],
  [D.g_share_total.toFixed(1) + "%", "GOOGL 等值敞口占比"],
  [D.div.ENB.toFixed(1), "有效独立赌注 ENB"],
  ["+" + D.sweep.be.toFixed(1) + "%/年", "HOLD 打平所需超额"],
].map(([n, l]) => `<div class="stat"><div class="num">${n}</div><div class="lbl">${l}</div></div>`).join("");

/* callouts */
document.getElementById("riskCallout").innerHTML =
  `风险分解结论：GOOGL 以 <b>${D.g_usd.toFixed(1)}%</b> 的资金占比贡献了 <b>${D.g_risk.toFixed(1)}%</b> 的组合方差；` +
  `前三大风险来源（GOOGL、QQQM、VINIX）合计 <b>${D.top3_risk.toFixed(1)}%</b>。组合方差的压缩路径由此清晰可见。`;
document.getElementById("simCallout").innerHTML =
  `<b>核心结论：</b>在期望收益相同的假设下，SELL 的终值中位数反而高于 HOLD` +
  `（${fmt$(D.sim.sell_med)} vs ${fmt$(D.sim.hold_med)}），其来源是<span class="term">波动率拖累</span>的结构性差异而非方向判断。` +
  `HOLD 的优势仅体现在 <span class="term">p95</span> 右侧尾部（+${fmt$k(D.sim.hold_p95 - D.sim.sell_p95)}），` +
  `代价为 <span class="term">p5</span> 左侧尾部恶化（-${fmt$k(D.sim.sell_p5 - D.sim.hold_p5)}）。` +
  `<b>仅当投资者确信 GOOG 具备年均 +${D.sweep.be.toFixed(1)}% 以上的持续超额（<span class="term">打平超额收益</span>）时，持有策略在中位数意义下方可成立。</b>`;

/* tables */
document.getElementById("ltTable").innerHTML =
  `<tr><th>口径</th><th>GOOGL 等值敞口</th><th>占净资产</th><th>GOOGL -30%</th><th>GOOGL -50%</th></tr>` +
  `<tr><td>流动账户</td><td>${fmt$(D.g)}</td><td>${D.g_share_liquid.toFixed(1)}%</td>` +
  `<td>-${fmt$k(D.shock.liquid[0])}</td><td>-${fmt$k(D.shock.liquid[1])}</td></tr>` +
  `<tr><td><b>含未归属 RSU</b></td><td><b>${fmt$(D.g + D.rsu)}</b></td><td><b>${D.g_share_total.toFixed(1)}%</b></td>` +
  `<td><b>-${fmt$k(D.shock.with_rsu[0])}</b></td><td><b>-${fmt$k(D.shock.with_rsu[1])}</b></td></tr>`;
document.getElementById("simTable").innerHTML =
  `<tr><th>策略</th><th>p5（不利）</th><th>中位数</th><th>p95（有利）</th><th>标准差</th><th>中位<span class="term">最大回撤</span></th></tr>` +
  `<tr><td>HOLD（继续持有 GOOG）</td><td>${fmt$(D.sim.hold_p5)}</td><td>${fmt$(D.sim.hold_med)}</td>` +
  `<td>${fmt$(D.sim.hold_p95)}</td><td>${fmt$(D.sim.hold_std)}</td><td>${D.sim.hold_dd_med.toFixed(1)}%</td></tr>` +
  `<tr><td>SELL（归属即卖转 <span class="term">VT</span>）</td><td>${fmt$(D.sim.sell_p5)}</td><td>${fmt$(D.sim.sell_med)}</td>` +
  `<td>${fmt$(D.sim.sell_p95)}</td><td>${fmt$(D.sim.sell_std)}</td><td>${D.sim.sell_dd_med.toFixed(1)}%</td></tr>`;
document.getElementById("evTable").innerHTML =
  `<tr><th>证据</th><th>数值</th><th>方法</th></tr>` +
  [
    ["GOOGL 等值敞口（含 RSU）", D.g_share_total.toFixed(1) + "%", "式 (2) 穿透计算"],
    ["市场因子 beta", "1.62 (t=18.9)", "式 (4) 因子回归"],
    ["有效独立赌注 ENB", D.div.ENB.toFixed(1), "式 (5) 特征值分解"],
    ["GOOGL 风险贡献", D.g_risk.toFixed(1) + "%", "式 (6) Euler 分解"],
    ["HOLD 打平所需超额", "+" + D.sweep.be.toFixed(1) + "%/年", "式 (12) 二分求解"],
  ].map(r => `<tr><td>${r[0]}</td><td><b>${r[1]}</b></td><td>${r[2]}</td></tr>`).join("");

/* charts */
if (!window.Plotly) {
  document.querySelectorAll(".chart").forEach(e => e.innerHTML =
    '<div class="fallback">图表组件加载失败。</div>');
} else {
  Plotly.newPlot("figComp", [{
    type:"pie", hole:.45, labels:D.sleeves.labels, values:D.sleeves.values,
    marker:{colors:D.sleeves.colors, line:{color:"#fff", width:2}},
    textinfo:"label+percent", hovertemplate:"%{label}<br>%{value:$,.0f} (%{percent})<extra></extra>",
    sort:false, direction:"clockwise", rotation:90, textfont:{size:12},
  }], {...LAY, title:{text:`资产配置结构（合计 ${fmt$(D.sleeves.total)}）`, font:{size:16}},
       showlegend:false, annotations:[{text:fmt$k(D.sleeves.total), showarrow:false, font:{size:20}}]}, CFG);

  let left = 0;
  const ltTraces = D.lookthrough.names.map((n, i) => {
    const tr = {type:"bar", orientation:"h", y:["GOOGL 等值敞口"], x:[D.lookthrough.values[i]],
                base:left, name:n, marker:{color:D.lookthrough.colors[i]},
                hovertemplate:`${n}: %{x:$,.0f}<extra></extra>`};
    left += D.lookthrough.values[i];
    return tr;
  });
  Plotly.newPlot("figLt", ltTraces, {...LAY, barmode:"stack", showlegend:true,
    legend:{orientation:"h", y:-0.25}, margin:{t:40, r:30, b:70, l:130},
    title:{text:`敞口占比：流动账户 ${D.g_share_liquid.toFixed(1)}% → 含 RSU ${D.g_share_total.toFixed(1)}%`, font:{size:15}},
    xaxis:{tickprefix:"$"}}, CFG);

  Plotly.newPlot("figShock", [
    {type:"bar", x:["GOOGL -30%", "GOOGL -50%"], y:D.shock.liquid, name:"仅流动账户",
     marker:{color:C.idx}, text:D.shock.liquid.map(v => "-" + fmt$k(v)), textposition:"outside"},
    {type:"bar", x:["GOOGL -30%", "GOOGL -50%"], y:D.shock.with_rsu, name:"含未归属 RSU",
     marker:{color:C.goog}, text:D.shock.with_rsu.map(v => "-" + fmt$k(v)), textposition:"outside"},
  ], {...LAY, barmode:"group", title:{text:"压力测试：账面损失（式 3）", font:{size:15}},
      legend:{orientation:"h", y:-0.2}, yaxis:{tickprefix:"$"}}, CFG);

  Plotly.newPlot("figFactor", [{
    type:"bar", x:D.factors.labels, y:D.factors.beta,
    marker:{color:D.factors.t.map(t => Math.abs(t) > 2 ? C.goog : C.grey)},
    customdata:D.factors.t,
    hovertemplate:"%{x}<br>beta=%{y:+.2f}<br>t=%{customdata:+.1f}<extra></extra>",
    text:D.factors.beta.map(b => b.toFixed(2)), textposition:"outside",
  }], {...LAY, title:{text:`因子 beta 估计（红色 = |t|>2 显著，R²=${D.r2.toFixed(0)}%）`, font:{size:15}},
       yaxis:{title:"beta", range:[Math.min(...D.factors.beta)-0.5, Math.max(...D.factors.beta)+0.6]}}, CFG);

  const rl = [...D.risk.labels].reverse(), rr = [...D.risk.risk].reverse(), ru = [...D.risk.usd].reverse();
  Plotly.newPlot("figRisk", [
    {type:"bar", orientation:"h", y:rl, x:rr, name:"风险贡献 %", marker:{color:C.goog}},
    {type:"bar", orientation:"h", y:rl, x:ru, name:"资金占比 %", marker:{color:C.idx}},
  ], {...LAY, barmode:"group", legend:{orientation:"h", y:-0.15},
      title:{text:`风险贡献 vs 资金占比（式 6）`, font:{size:15}}}, CFG);

  Plotly.newPlot("figSim", [
    {type:"bar", x:D.sim.centers, y:D.sim.hold, name:"HOLD 持有 GOOG",
     marker:{color:C.goog, opacity:.65}, hovertemplate:"终值 %{x:$,.0f}<br>路径数 %{y}<extra>HOLD</extra>"},
    {type:"bar", x:D.sim.centers, y:D.sim.sell, name:"SELL 转 VT",
     marker:{color:C.idx, opacity:.65}, hovertemplate:"终值 %{x:$,.0f}<br>路径数 %{y}<extra>SELL</extra>"},
  ], {...LAY, barmode:"overlay", legend:{orientation:"h", y:-0.18},
      title:{text:"48 个月期终值经验分布（虚线 = 中位数，20,000 条路径）", font:{size:15}}, xaxis:{tickprefix:"$"},
      shapes:[
        {type:"line", x0:D.sim.hold_med, x1:D.sim.hold_med, y0:0, y1:1, yref:"paper",
         line:{color:C.goog, dash:"dash", width:2}},
        {type:"line", x0:D.sim.sell_med, x1:D.sim.sell_med, y0:0, y1:1, yref:"paper",
         line:{color:C.idx, dash:"dash", width:2}},
      ]}, CFG);

  Plotly.newPlot("figSweep", [{
    type:"scatter", mode:"lines", x:D.sweep.x, y:D.sweep.y, name:"HOLD − SELL 中位数差",
    line:{color:C.ink, width:2.5}, fill:"tozeroy", fillcolor:"rgba(181,68,42,.08)",
    hovertemplate:"超额 %{x}%/年 → 中位数差 %{y:$,.0f}<extra></extra>",
  }, {
    type:"scatter", mode:"markers", x:[4.5], y:[0], name:"情景假设",
    marker:{size:13, color:C.goog, symbol:"diamond"},
  }], {...LAY, showlegend:false, title:{text:"HOLD − SELL 终值中位数差对超额收益假设的敏感性", font:{size:15}},
       xaxis:{title:"GOOG 年均超额收益假设（%/年）", ticksuffix:"%"},
       yaxis:{title:"中位数差（美元）", tickprefix:"$", zeroline:true,
              zerolinecolor:"#999", zerolinewidth:1.5},
       annotations:[{x:D.sweep.be, y:0, text:`打平点 +${D.sweep.be.toFixed(1)}%`,
                     showarrow:true, arrowhead:2, ax:0, ay:-40, font:{color:C.goog}}]}, CFG);

  const slider = document.getElementById("drift");
  function upd() {
    const x = parseFloat(slider.value);
    document.getElementById("driftLbl").textContent = "+" + x.toFixed(1) + "%/年";
    let y = NaN;
    for (let i = 0; i < D.sweep.x.length - 1; i++) {
      if (x >= D.sweep.x[i] && x <= D.sweep.x[i+1]) {
        const f = (x - D.sweep.x[i]) / (D.sweep.x[i+1] - D.sweep.x[i]);
        y = D.sweep.y[i] + f * (D.sweep.y[i+1] - D.sweep.y[i]);
        break;
      }
    }
    if (isNaN(y)) y = D.sweep.y[D.sweep.y.length - 1];
    Plotly.update("figSweep", {x:[[x]], y:[[y]]}, {}, [1]);
    const v = document.getElementById("verdict");
    if (y >= 0) {
      v.innerHTML = `该假设下 HOLD 中位数终值较 SELL <span class="good">高 ${fmt$k(y)}</span>，持有策略在中位数意义下成立`;
    } else {
      v.innerHTML = `该假设下 HOLD 中位数终值较 SELL <span class="bad">低 ${fmt$k(-y)}</span>，卖出分散在中位数意义下占优`;
    }
  }
  slider.addEventListener("input", upd);
  upd();
}

/* term popovers */
const pop = document.getElementById("popover");
function baseName(k) { return k.split("（")[0].split("(")[0].trim(); }
function bindTerms() {
  document.querySelectorAll(".term").forEach(el => {
    el.addEventListener("click", ev => {
      const name = el.textContent;
      let key = Object.keys(TERMS).find(k => name.includes(k) || k.includes(name));
      if (!key) key = Object.keys(TERMS).find(k => baseName(k) === name ||
                     name.includes(baseName(k)));
      if (!key) return;
      pop.querySelector(".pt").textContent = key;
      pop.querySelector(".pd").textContent = TERMS[key];
      pop.style.display = "block";
      const r = el.getBoundingClientRect();
      pop.style.left = Math.min(r.left, window.innerWidth - 360) + "px";
      pop.style.top = (r.bottom + window.scrollY + 8) + "px";
      ev.stopPropagation();
    });
  });
}
document.addEventListener("click", () => pop.style.display = "none");
bindTerms();
</script>
</body>
</html>
"""


def main():
    print("collecting data (prices / weights / calibration / sweep)...")
    d = gather()
    data = build_data(d)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(build_html(data))
    print(f"HTML written: {OUT_HTML}")


if __name__ == "__main__":
    main()
