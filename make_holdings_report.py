"""Generate a live holdings report webpage (holdings_report.html).

Reads profolio.xlsx, fetches live prices via yfinance, renders a single-file
interactive HTML: totals, day change, per-account and per-symbol views.

Notes:
- VINIX/VIEIX are 401(k) institutional trusts; yfinance NAVs for the
  similarly-named mutual funds are wrong, so sheet prices are used.
- M1 rows lack quantity; qty is inferred from market value / live price.
- Unvested RSU (from config_local.UNVESTED_RSU_SHARES) shown as memo only.

Usage: .venv/bin/python make_holdings_report.py
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

OUT = "holdings_report.html"
CASH_SYMBOLS = {"SPAXX", "FDRXX", "QPCTQ", "CASH", "VMFXX"}
TRUST_SHEET_PRICE = {"VINIX", "VIEIX"}  # price from sheet, not yfinance

try:
    from config_local import UNVESTED_RSU_SHARES
except ImportError:
    UNVESTED_RSU_SHARES = 0.0


def _js(path):
    with open(path, encoding="utf-8") as f:
        return f.read().replace("</script", "<\\/script")


def gather():
    df = pd.read_excel("profolio.xlsx", sheet_name="Holdings Detail")
    df = df[df["Symbol"].notna()]
    df = df[~df["Name"].astype(str).str.upper().str.contains("TOTAL")].copy()
    mv = pd.to_numeric(df["Market Value"], errors="coerce")
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    px_sheet = pd.to_numeric(df["Current Price"], errors="coerce")
    df["sheet_mv"] = mv.fillna(qty * px_sheet)
    df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip().str.replace("*", "", regex=False)

    tickers = sorted(s for s in df["Symbol"].unique()
                     if s not in CASH_SYMBOLS and s not in TRUST_SHEET_PRICE)
    hist = yf.download(tickers, period="10d", interval="1d", progress=False,
                       threads=False)["Close"].ffill()
    live = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else hist.iloc[-1]
    asof = hist.index[-1].date().isoformat()

    rows = []
    for _, r in df.iterrows():
        sym = r["Symbol"]
        sheet_mv = float(r["sheet_mv"])
        gain_sheet = pd.to_numeric(pd.Series([r.get("Unrealized Gain $")]), errors="coerce").iloc[0]
        gain_sheet = 0.0 if pd.isna(gain_sheet) else float(gain_sheet)

        if sym in CASH_SYMBOLS:
            rows.append(dict(account=r["Account"], sym=sym, name=str(r["Name"]),
                             qty=None, cost_px=None, live_px=1.0, prev_px=1.0,
                             mv=sheet_mv, cost=sheet_mv, gain=0.0,
                             price_src="cash", cash=True))
            continue

        if sym in TRUST_SHEET_PRICE:
            p = float(r["Current Price"])
            q = float(r["Quantity"])
            rows.append(dict(account=r["Account"], sym=sym, name=str(r["Name"]),
                             qty=q, cost_px=p, live_px=p, prev_px=p,
                             mv=sheet_mv, cost=sheet_mv, gain=0.0,
                             price_src="sheet", cash=False))
            continue

        p = float(live[sym])
        pp = float(prev[sym])
        if pd.notna(r["Quantity"]):
            q = float(r["Quantity"])
            src = "live"
        else:  # M1 rows: infer qty from sheet market value
            q = sheet_mv / p
            src = "live(qty inferred)"
        if pd.notna(r["Avg Cost"]):
            cost = q * float(r["Avg Cost"])
            cost_px = float(r["Avg Cost"])
        else:
            cost = sheet_mv - gain_sheet  # e.g. M1 rows, GOOGL RSU (gain 0)
            cost_px = cost / q if q else None
        rows.append(dict(account=r["Account"], sym=sym, name=str(r["Name"]),
                         qty=q, cost_px=cost_px, live_px=p, prev_px=pp,
                         mv=q * p, cost=cost, gain=q * p - cost,
                         price_src=src, cash=False))
    return rows, asof


def build(rows, asof):
    positions = [r for r in rows if not r["cash"]]
    cash_total = sum(r["mv"] for r in rows if r["cash"])
    mv_total = sum(r["mv"] for r in positions)
    cost_total = sum(r["cost"] for r in positions)
    gain_total = mv_total - cost_total
    day_total = sum(r["qty"] * (r["live_px"] - r["prev_px"]) for r in positions)

    by_sym = {}
    for r in positions:
        s = by_sym.setdefault(r["sym"], dict(sym=r["sym"], name=r["name"], qty=0.0,
                                             mv=0.0, cost=0.0, day=0.0,
                                             live_px=r["live_px"], accounts=set()))
        s["qty"] += r["qty"]
        s["mv"] += r["mv"]
        s["cost"] += r["cost"]
        s["day"] += r["qty"] * (r["live_px"] - r["prev_px"])
        s["accounts"].add(r["account"])
    syms = sorted(by_sym.values(), key=lambda x: -x["mv"])

    by_acc = {}
    for r in rows:
        a = by_acc.setdefault(r["account"], dict(account=r["account"], mv=0.0,
                                                 cost=0.0, day=0.0, n=0))
        a["mv"] += r["mv"]
        a["cost"] += r["cost"]
        a["n"] += 1
        if not r["cash"]:
            a["day"] += r["qty"] * (r["live_px"] - r["prev_px"])
    accs = sorted(by_acc.values(), key=lambda x: -x["mv"])

    rsu_mv = UNVESTED_RSU_SHARES * float(
        next((r["live_px"] for r in positions if r["sym"] == "GOOGL"), 0.0))

    return dict(
        asof=asof, generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
        mv_total=mv_total, cost_total=cost_total, gain_total=gain_total,
        gain_pct=(gain_total / cost_total * 100) if cost_total else 0.0,
        day_total=day_total, cash_total=cash_total,
        grand_total=mv_total + cash_total,
        rsu_shares=UNVESTED_RSU_SHARES, rsu_mv=rsu_mv,
        syms=[{**s, "accounts": sorted(s["accounts"]),
               "gain": s["mv"] - s["cost"],
               "gain_pct": (s["mv"] - s["cost"]) / s["cost"] * 100 if s["cost"] else 0.0}
              for s in syms],
        accs=[{**a, "gain": a["mv"] - a["cost"]} for a in accs],
        rows=positions,
    )


def _perf_metrics(r, rf=0.04):
    """Risk/return metrics from a daily simple-return series."""
    n = len(r)
    curve = (1 + r).cumprod()
    cum = float(curve.iloc[-1] - 1)
    ann = float(curve.iloc[-1] ** (252 / n) - 1)
    vol = float(r.std() * np.sqrt(252))
    dd = float(((curve.cummax() - curve) / curve.cummax()).max())
    downside = r[r < 0]
    dvol = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else float("nan")
    return dict(cum=cum * 100, ann=ann * 100, vol=vol * 100,
                sharpe=(ann - rf) / vol if vol else float("nan"),
                sortino=(ann - rf) / dvol if dvol else float("nan"),
                max_dd=dd * 100, calmar=ann / dd if dd else float("nan"))


def _rel_metrics(rp, rb):
    """Portfolio-vs-benchmark relative metrics from aligned daily returns."""
    idx = rp.index.intersection(rb.index)
    a, b = rp.loc[idx].values, rb.loc[idx].values
    beta, alpha_d = np.polyfit(b, a, 1)
    r2 = float(np.corrcoef(a, b)[0, 1] ** 2)
    te = float(np.std(a - b, ddof=1) * np.sqrt(252))
    alpha_ann = float(alpha_d * 252)
    up, dn = b > 0, b < 0
    return dict(beta=float(beta), alpha=alpha_ann * 100, r2=r2, te=te * 100,
                ir=alpha_ann / te if te else float("nan"),
                up_cap=float(a[up].mean() / b[up].mean()) if up.any() else float("nan"),
                dn_cap=float(a[dn].mean() / b[dn].mean()) if dn.any() else float("nan"))


def compute_performance(rows, cash_total):
    """Backtest the current (unchanged) portfolio over the past year.

    Uses dividend/split-adjusted closes. The 401(k) trusts (VINIX/VIEIX) have
    no usable NAV history, so the mutual fund's price *return* is applied to
    the trust's sheet price — same underlying index.
    """
    positions = [r for r in rows if not r["cash"]]
    qty = {}
    for r in positions:
        qty[r["sym"]] = qty.get(r["sym"], 0.0) + r["qty"]
    syms = sorted(qty)
    px = yf.download(syms, period="400d", interval="1d", progress=False,
                     threads=False, auto_adjust=True)["Close"].ffill()
    if isinstance(px, pd.Series):
        px = px.to_frame(syms[0])
    # bfill: for pre-inception history (e.g. DRAM launched 2026-04) hold the
    # first available price flat, i.e. assume zero return before listing.
    short_hist = [c for c in px.columns if px[c].first_valid_index() is not None
                  and px[c].first_valid_index() > px.index[0]]
    px = px.bfill()

    sheet_px = {r["sym"]: r["live_px"] for r in positions}
    V = pd.Series(cash_total, index=px.index)
    for s, q in qty.items():
        series = px[s]
        if s in TRUST_SHEET_PRICE:
            series = series / series.iloc[-1] * sheet_px[s]
        V = V + q * series
    V = V.dropna().tail(253)

    def at(days=None, months=None, ytd=False):
        if ytd:
            t = pd.Timestamp(V.index[-1].year, 1, 1)
            idx = V.index[V.index >= t]
            return V.loc[idx[0]] if len(idx) else V.iloc[0]
        t = V.index[-1] - (pd.Timedelta(days=days) if days else pd.DateOffset(months=months))
        idx = V.index[V.index <= t]
        return V.loc[idx[-1]] if len(idx) else V.iloc[0]

    now = V.iloc[-1]
    periods = []
    for label, v0 in [("1W", at(days=7)), ("1M", at(months=1)), ("3M", at(months=3)),
                      ("YTD", at(ytd=True)), ("1Y", at(days=365))]:
        periods.append(dict(label=label, pct=float(now / v0 - 1) * 100,
                            usd=float(now - v0), v0=float(v0)))

    # benchmarks: SPY = broad market, QQQ = de-facto exposure, SMH = semi tilt
    bpx = yf.download(["SPY", "QQQ", "SMH"], period="400d", interval="1d",
                      progress=False, threads=False, auto_adjust=True)["Close"].ffill()
    rp = V.pct_change().dropna()

    def curves(series):
        r = series.pct_change().dropna()
        c = (1 + r).cumprod()
        dd = (c.cummax() - c) / c.cummax()
        roll = r.rolling(60).std() * np.sqrt(252) * 100
        return (r, [round(float(x * 100), 2) for x in dd.values],
                [None if pd.isna(x) else round(float(x), 1) for x in roll.values])

    bench = {}
    for b in ["SPY", "QQQ", "SMH"]:
        bs = bpx[b].reindex(V.index).ffill()
        rb, dd_c, roll_c = curves(bs)
        bench[b] = dict(m=_perf_metrics(rb), rel=_rel_metrics(rp, rb),
                        norm=[round(float(v / bs.iloc[0] * 100), 2) for v in bs.values],
                        dd=dd_c, roll60=roll_c,
                        ret_1y=float(bs.iloc[-1] / bs.iloc[0] - 1) * 100)

    rp_dd, rp_roll = curves(V)[1:]
    pm = _perf_metrics(rp)
    dd = pm["max_dd"]
    vol = pm["vol"]

    # verdict: decompose excess over QQQ into beta-expected and alpha parts
    q = bench["QQQ"]
    beta_expected = q["rel"]["beta"] * q["ret_1y"]
    alpha_part = pm["cum"] - beta_expected
    eff = ("超额收益并未以牺牲风险调整效率为代价"
           if pm["sharpe"] >= q["m"]["sharpe"]
           else "超额收益伴随风险调整效率的下降")
    verdict = (
        f"近一年组合累计 {pm['cum']:+.1f}%（SPY {bench['SPY']['ret_1y']:+.1f}%、"
        f"QQQ {q['ret_1y']:+.1f}%、SMH {bench['SMH']['ret_1y']:+.1f}%）。"
        f"对 QQQ 的 beta 为 {q['rel']['beta']:.2f}——若组合只是 QQQ 的杠杆复制，"
        f"预期收益约 {beta_expected:+.1f}%；实际超出 {alpha_part:+.1f}%，"
        f"回归口径年化 alpha {q['rel']['alpha']:+.1f}%（信息比率 {q['rel']['ir']:.2f}，"
        f"R²={q['rel']['r2']:.2f}）。Sharpe {pm['sharpe']:.2f} vs QQQ {q['m']['sharpe']:.2f} "
        f"vs SMH {bench['SMH']['m']['sharpe']:.2f}；最大回撤 {dd:.1f}% vs "
        f"QQQ {q['m']['max_dd']:.1f}% vs SMH {bench['SMH']['m']['max_dd']:.1f}%。{eff}。")

    return dict(
        dates=[d.date().isoformat() for d in V.index],
        port=[round(float(v)) for v in V.values],
        port_norm=[round(float(v / V.iloc[0] * 100), 2) for v in V.values],
        periods=periods, max_dd=dd, vol=vol,
        bench=bench, port_m=pm, port_dd=rp_dd, port_roll60=rp_roll,
        spy_norm=bench["SPY"]["norm"], bench_max_dd=bench["SPY"]["m"]["max_dd"],
        spy_1y=bench["SPY"]["ret_1y"],
        short_hist=short_hist, verdict=verdict,
    )


HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>持仓报告 · Live</title>
<script>%%PLOTLY%%</script>
<style>
:root { --ink:#1f2430; --muted:#5d6673; --bg:#f7f6f3; --card:#fff; --line:#e5e2db;
        --red:#b5442a; --green:#2f7d57; --idx:#3f5f8f; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:-apple-system,"PingFang SC","Hiragino Sans GB",sans-serif;
       background:var(--bg); color:var(--ink); line-height:1.7; }
.wrap { max-width:1080px; margin:0 auto; padding:24px 20px 60px; }
header { display:flex; justify-content:space-between; align-items:baseline;
         border-bottom:3px solid var(--ink); padding-bottom:12px; margin-bottom:20px; }
h1 { font-size:24px; letter-spacing:2px; }
.meta { color:var(--muted); font-size:12.5px; text-align:right; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:18px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:14px 18px; }
.stat .n { font-size:21px; font-weight:600; }
.stat .l { font-size:12px; color:var(--muted); }
.pos { color:var(--green); } .neg { color:var(--red); }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px;
        padding:18px 22px; margin:16px 0; }
.tabs { margin:8px 0 0; }
.tabs button { border:1px solid var(--line); background:#f0eee9; color:var(--muted);
  padding:6px 18px; border-radius:6px; cursor:pointer; font-size:13.5px; margin-right:6px; }
.tabs button.active { background:var(--ink); color:#fff; border-color:var(--ink); }
table { width:100%; border-collapse:collapse; font-size:13.5px; }
th { background:var(--ink); color:#fff; padding:8px 10px; text-align:right; font-weight:600;
     position:sticky; top:0; }
th:first-child, td:first-child { text-align:left; }
td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:right;
     font-variant-numeric:tabular-nums; }
tr:nth-child(even) td { background:#faf8f4; }
tr.total td { font-weight:700; background:#f0eee9; border-top:2px solid var(--ink); }
th.sortable { cursor:pointer; user-select:none; }
th.sortable:hover { background:#2c3342; }
th.sortable::after { content:" ⇅"; opacity:.45; font-size:11px; }
th.sortable[data-dir="asc"]::after { content:" ↑"; opacity:1; }
th.sortable[data-dir="desc"]::after { content:" ↓"; opacity:1; }
.tblwrap { max-height:560px; overflow-y:auto; }
.chart { width:100%; height:420px; }
.note { color:var(--muted); font-size:12.5px; margin-top:10px; line-height:1.9; }
.memo { background:#eef2f7; border-left:3px solid var(--idx); border-radius:4px;
        padding:12px 18px; margin:14px 0; font-size:14px; }
.badge { font-size:11px; color:var(--muted); border:1px solid var(--line);
         border-radius:4px; padding:0 5px; margin-left:6px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>持仓报告</h1>
    <div class="meta" id="meta"></div>
  </header>
  <div class="stats" id="stats"></div>
  <div class="memo" id="rsu"></div>
  <div class="card">
    <h3 style="margin-bottom:10px">区间表现（假设持仓不变，含股息调整）</h3>
    <div class="stats" id="perf" style="margin:4px 0 10px"></div>
    <div id="figPerf" class="chart" style="height:380px"></div>
    <div class="note" id="perfNote"></div>
  </div>
  <div class="card">
    <h3 style="margin-bottom:10px">基准对比（近一年） · SPY / QQQ / SMH</h3>
    <div class="memo" id="verdict"></div>
    <div class="tblwrap" style="max-height:none"><table id="tblBase"></table></div>
    <div class="tblwrap" style="max-height:none; margin-top:14px"><table id="tblRel"></table></div>
    <div id="figDD" class="chart" style="height:320px; margin-top:14px"></div>
    <div id="figRoll" class="chart" style="height:320px"></div>
    <div class="note">方法：Sharpe / Sortino 以 4% 无风险利率计；beta 与年化 alpha 为组合日收益对基准日收益的单因子 OLS 回归（alpha = 日截距 × 252）；跟踪误差为超额收益日序列的年化标准差，信息比率 = 年化 alpha ÷ 跟踪误差；上行/下行捕获 = 基准上涨/下跌日中组合日均收益与基准日均收益之比。三者角色：SPY = 宽基躺平基准，QQQ = 组合实际敞口的最近影子，SMH = 半导体倾斜对照。</div>
  </div>
  <div class="card"><div id="figAlloc" class="chart"></div></div>
  <div class="card"><div id="figGain" class="chart" style="height:380px"></div></div>
  <div class="card">
    <div class="tabs">
      <button id="btnSym" class="active" onclick="showTab('sym')">按标的合并</button>
      <button id="btnAcc" onclick="showTab('acc')">按账户</button>
      <button id="btnLot" onclick="showTab('lot')">逐笔明细</button>
    </div>
    <div class="tblwrap">
      <table id="tblSym"></table>
      <table id="tblAcc" style="display:none"></table>
      <table id="tblLot" style="display:none"></table>
    </div>
    <div class="note" id="notes"></div>
  </div>
</div>
<script>
const D = %%DATA%%;
const fmt$ = (x,d=0) => (x<0?"-":"") + "$" + Math.abs(x).toLocaleString("en-US",{minimumFractionDigits:d,maximumFractionDigits:d});
const fmtPct = x => (x>=0?"+":"") + x.toFixed(1) + "%";
const cls = x => x>=0 ? "pos" : "neg";
const sgn = (x,d=0) => (x>=0?"+":"") + fmt$(x,d).replace("$","$").replace("-","-").replace(/^/, x>=0?"":"-").replace("--","-");
const money = x => (x>=0?"+":"−") + "$" + Math.abs(x).toLocaleString("en-US",{maximumFractionDigits:0});

document.getElementById("meta").innerHTML =
  `行情日期：${D.asof}（收盘价）<br>生成时间：${D.generated}`;
document.getElementById("stats").innerHTML = [
  [fmt$(D.grand_total), "总资产（含现金）", ""],
  [fmt$(D.mv_total), "证券市值", ""],
  [fmt$(D.cash_total), "现金", ""],
  [`<span class="${cls(D.gain_total)}">${money(D.gain_total)} (${fmtPct(D.gain_pct)})</span>`, "未实现盈亏", ""],
  [`<span class="${cls(D.day_total)}">${money(D.day_total)}</span>`, "最近一个交易日变动", ""],
].map(([n,l]) => `<div class="stat"><div class="n">${n}</div><div class="l">${l}</div></div>`).join("");

document.getElementById("rsu").innerHTML =
  `<b>备忘（不计入上方合计）：</b>未归属 RSU ${D.rsu_shares.toLocaleString()} 股 GOOGL，` +
  `按现价折算约 <b>${fmt$(D.rsu_mv)}</b>。归属前不可交易，且价值同时受股价与在职状态影响。`;

/* performance section */
document.getElementById("perf").innerHTML =
  D.perf.periods.map(p =>
    `<div class="stat"><div class="n ${cls(p.pct)}">${fmtPct(p.pct)}</div>` +
    `<div class="l">${p.label} · ${money(p.usd)}</div></div>`).join("") +
  `<div class="stat"><div class="n neg">-${D.perf.max_dd.toFixed(1)}%</div>` +
  `<div class="l">1Y 最大回撤（SPY ${D.perf.bench_max_dd.toFixed(1)}%）</div></div>` +
  `<div class="stat"><div class="n">${D.perf.vol.toFixed(1)}%</div>` +
  `<div class="l">1Y 年化波动率</div></div>`;
document.getElementById("perfNote").innerHTML =
  `过去一年组合 ${fmtPct(D.perf.periods[4].pct)}，同期 SPY ${fmtPct(D.perf.spy_1y)}。` +
  `VINIX/VIEIX 信托用同名基金的指数收益做代理；现金按零收益处理` +
  (D.perf.short_hist.length
    ? `；${D.perf.short_hist.join("/")} 上市不足一年，上市前收益按 0 处理。`
    : "。");

/* benchmark comparison */
document.getElementById("verdict").innerHTML = "<b>结论：</b>" + D.perf.verdict;
(function () {
  const P = D.perf, B = P.bench, pm = P.port_m;
  const rows1 = [
    ["累计收益", fmtPct(pm.cum), fmtPct(B.SPY.m.cum), fmtPct(B.QQQ.m.cum), fmtPct(B.SMH.m.cum)],
    ["年化波动率", pm.vol.toFixed(1)+"%", B.SPY.m.vol.toFixed(1)+"%", B.QQQ.m.vol.toFixed(1)+"%", B.SMH.m.vol.toFixed(1)+"%"],
    ["Sharpe", pm.sharpe.toFixed(2), B.SPY.m.sharpe.toFixed(2), B.QQQ.m.sharpe.toFixed(2), B.SMH.m.sharpe.toFixed(2)],
    ["Sortino", pm.sortino.toFixed(2), B.SPY.m.sortino.toFixed(2), B.QQQ.m.sortino.toFixed(2), B.SMH.m.sortino.toFixed(2)],
    ["最大回撤", "-"+pm.max_dd.toFixed(1)+"%", "-"+B.SPY.m.max_dd.toFixed(1)+"%", "-"+B.QQQ.m.max_dd.toFixed(1)+"%", "-"+B.SMH.m.max_dd.toFixed(1)+"%"],
    ["Calmar", pm.calmar.toFixed(2), B.SPY.m.calmar.toFixed(2), B.QQQ.m.calmar.toFixed(2), B.SMH.m.calmar.toFixed(2)],
  ];
  document.getElementById("tblBase").innerHTML =
    `<tr><th>指标</th><th>组合</th><th>SPY</th><th>QQQ</th><th>SMH</th></tr>` +
    rows1.map(r => `<tr><td>${r[0]}</td><td><b>${r[1]}</b></td><td>${r[2]}</td><td>${r[3]}</td><td>${r[4]}</td></tr>`).join("");
  const rl = [B.SPY.rel, B.QQQ.rel, B.SMH.rel];
  const rows2 = [
    ["beta", ...rl.map(x => x.beta.toFixed(2))],
    ["年化 alpha", ...rl.map(x => fmtPct(x.alpha))],
    ["R²", ...rl.map(x => x.r2.toFixed(2))],
    ["跟踪误差", ...rl.map(x => x.te.toFixed(1)+"%")],
    ["信息比率", ...rl.map(x => x.ir.toFixed(2))],
    ["上行捕获", ...rl.map(x => x.up_cap.toFixed(2))],
    ["下行捕获", ...rl.map(x => x.dn_cap.toFixed(2))],
  ];
  document.getElementById("tblRel").innerHTML =
    `<tr><th>相对指标（组合 vs 基准）</th><th>vs SPY</th><th>vs QQQ</th><th>vs SMH</th></tr>` +
    rows2.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("");
})();

/* symbol table */
let rows = D.syms.map(s => {
  const gp = s.gain_pct;
  return `<tr><td><b>${s.sym}</b><span class="badge">${s.accounts.length} 账户</span></td>` +
    `<td>${s.qty.toLocaleString("en-US",{maximumFractionDigits:2})}</td>` +
    `<td>${fmt$(s.live_px,2)}</td><td>${fmt$(s.mv)}</td>` +
    `<td class="${cls(s.day)}">${money(s.day)}</td>` +
    `<td class="${cls(s.gain)}">${money(s.gain)}</td>` +
    `<td class="${cls(s.gain)}">${fmtPct(gp)}</td>` +
    `<td>${(s.mv/D.grand_total*100).toFixed(1)}%</td></tr>`;
}).join("");
document.getElementById("tblSym").innerHTML =
  `<tr><th>标的</th><th>股数</th><th>现价</th><th>市值</th><th>当日变动</th><th>未实现盈亏</th><th>盈亏%</th><th>占比</th></tr>` + rows +
  `<tr class="total"><td>现金合计</td><td></td><td></td><td>${fmt$(D.cash_total)}</td><td></td><td></td><td></td>` +
  `<td>${(D.cash_total/D.grand_total*100).toFixed(1)}%</td></tr>` +
  `<tr class="total"><td>总计</td><td></td><td></td><td>${fmt$(D.grand_total)}</td><td class="${cls(D.day_total)}">${money(D.day_total)}</td>` +
  `<td class="${cls(D.gain_total)}">${money(D.gain_total)}</td><td class="${cls(D.gain_total)}">${fmtPct(D.gain_pct)}</td><td>100%</td></tr>`;

/* account table */
document.getElementById("tblAcc").innerHTML =
  `<tr><th>账户</th><th>头寸数</th><th>市值（含现金）</th><th>当日变动</th><th>未实现盈亏</th><th>占比</th></tr>` +
  D.accs.map(a => `<tr><td>${a.account}</td><td>${a.n}</td><td>${fmt$(a.mv)}</td>` +
    `<td class="${cls(a.day)}">${money(a.day)}</td>` +
    `<td class="${cls(a.gain)}">${money(a.gain)}</td>` +
    `<td>${(a.mv/D.grand_total*100).toFixed(1)}%</td></tr>`).join("") +
  `<tr class="total"><td>总计</td><td></td><td>${fmt$(D.grand_total)}</td>` +
  `<td class="${cls(D.day_total)}">${money(D.day_total)}</td><td class="${cls(D.gain_total)}">${money(D.gain_total)}</td><td>100%</td></tr>`;

/* lot table */
document.getElementById("tblLot").innerHTML =
  `<tr><th>账户</th><th>标的</th><th>股数</th><th>成本价</th><th>现价</th><th>市值</th><th>未实现盈亏</th></tr>` +
  D.rows.map(r => `<tr><td>${r.account}</td><td><b>${r.sym}</b></td>` +
    `<td>${r.qty.toLocaleString("en-US",{maximumFractionDigits:3})}</td>` +
    `<td>${r.cost_px==null?"—":fmt$(r.cost_px,2)}</td><td>${fmt$(r.live_px,2)}</td>` +
    `<td>${fmt$(r.mv)}</td><td class="${cls(r.gain)}">${money(r.gain)}</td></tr>`).join("") +
  D.rows_cash.map(r => `<tr><td>${r.account}</td><td>${r.sym}</td><td></td><td></td><td></td>` +
    `<td>${fmt$(r.mv)}</td><td></td></tr>`).join("");

function showTab(t) {
  for (const k of ["Sym","Acc","Lot"]) {
    document.getElementById("tbl"+k).style.display = k.toLowerCase()===t ? "" : "none";
    document.getElementById("btn"+k).className = k.toLowerCase()===t ? "active" : "";
  }
}

/* sortable columns: click header to toggle asc/desc; total rows stay pinned */
function parseCell(cell) {
  let t = cell.querySelector(".badge") ? cell.childNodes[0].textContent : cell.textContent;
  t = t.trim();
  if (t === "" || t === "—") return -Infinity;
  const n = parseFloat(t.replace(/[$,%\s,]/g, "").replace(/−/g, "-").replace(/\+/g, ""));
  return isNaN(n) ? t : n;
}
function makeSortable(id) {
  const tbl = document.getElementById(id);
  const ths = Array.from(tbl.querySelectorAll("th"));
  ths.forEach((th, col) => {
    th.classList.add("sortable");
    th.addEventListener("click", () => {
      const asc = th.dataset.dir !== "asc";
      ths.forEach(h => delete h.dataset.dir);
      th.dataset.dir = asc ? "asc" : "desc";
      const all = Array.from(tbl.querySelectorAll("tr"));
      const header = all[0];
      const data = all.slice(1).filter(r => !r.classList.contains("total"));
      const totals = all.slice(1).filter(r => r.classList.contains("total"));
      data.sort((a, b) => {
        const x = parseCell(a.cells[col]), y = parseCell(b.cells[col]);
        const cmp = (typeof x === "number" && typeof y === "number")
          ? x - y : String(x).localeCompare(String(y), "zh-CN");
        return asc ? cmp : -cmp;
      });
      tbl.innerHTML = "";
      tbl.appendChild(header);
      data.forEach(r => tbl.appendChild(r));
      totals.forEach(r => tbl.appendChild(r));
    });
  });
}
["tblSym", "tblAcc", "tblLot"].forEach(makeSortable);

/* charts */
const LAY = {font:{family:"-apple-system, PingFang SC, sans-serif", color:"#1f2430"},
             margin:{t:44, r:20, b:40, l:60}, paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)"};
const CFG = {responsive:true, displayModeBar:false};
if (window.Plotly) {
  const BC = {SPY:"#3f5f8f", QQQ:"#6b6391", SMH:"#a8903a"};
  const normTraces = [
    {type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.port_norm, name:"本组合",
     line:{color:"#b5442a", width:2.4}, hovertemplate:"%{x}<br>组合 %{y:.1f}<extra></extra>"},
    ...["SPY","QQQ","SMH"].map(b => ({
      type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.bench[b].norm, name:b,
      line:{color:BC[b], width:1.4, dash:b==="SPY"?"dot":"solid"},
      hovertemplate:`%{x}<br>${b} %{y:.1f}<extra></extra>`})),
  ];
  Plotly.newPlot("figPerf", normTraces,
    {...LAY, title:{text:"近一年净值走势（期初 = 100，每日数据；拖动底部滑条缩放）"},
     legend:{orientation:"h", y:-0.18},
     xaxis:{dtick:7*24*3600*1000, tickformat:"%m-%d",
            rangeslider:{visible:true, thickness:0.09}},
     yaxis:{title:"净值", fixedrange:false}}, CFG);

  Plotly.newPlot("figDD", [
    {type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.port_dd.map(v=>-v), name:"本组合",
     line:{color:"#b5442a", width:2}, fill:"tozeroy", fillcolor:"rgba(181,68,42,.12)",
     hovertemplate:"%{x}<br>组合 %{y:.1f}%<extra></extra>"},
    ...["SPY","QQQ","SMH"].map(b => ({
      type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.bench[b].dd.map(v=>-v), name:b,
      line:{color:BC[b], width:1.3}, hovertemplate:`%{x}<br>${b} %{y:.1f}%<extra></extra>`})),
  ], {...LAY, title:{text:"水下回撤曲线"}, legend:{orientation:"h", y:-0.2},
      yaxis:{title:"回撤", ticksuffix:"%"}}, CFG);

  Plotly.newPlot("figRoll", [
    {type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.port_roll60, name:"本组合",
     line:{color:"#b5442a", width:2}, hovertemplate:"%{x}<br>组合 %{y:.1f}%<extra></extra>"},
    ...["SPY","QQQ","SMH"].map(b => ({
      type:"scatter", mode:"lines", x:D.perf.dates, y:D.perf.bench[b].roll60, name:b,
      line:{color:BC[b], width:1.3}, hovertemplate:`%{x}<br>${b} %{y:.1f}%<extra></extra>`})),
  ], {...LAY, title:{text:"滚动 60 日年化波动率"}, legend:{orientation:"h", y:-0.2},
      yaxis:{title:"波动率", ticksuffix:"%"}}, CFG);

  const top = D.syms.slice(0, 10);
  const other = D.syms.slice(10);
  const labels = top.map(s=>s.sym), values = top.map(s=>s.mv);
  if (other.length) { labels.push("其它+现金"); values.push(other.reduce((a,s)=>a+s.mv,0)+D.cash_total); }
  Plotly.newPlot("figAlloc", [{
    type:"pie", hole:.45, labels, values, sort:false, direction:"clockwise", rotation:90,
    textinfo:"label+percent", hovertemplate:"%{label}<br>%{value:$,.0f} (%{percent})<extra></extra>",
    marker:{line:{color:"#fff",width:2}},
  }], {...LAY, title:{text:"配置（按标的，含现金）"}, showlegend:false,
       annotations:[{text:fmt$(D.grand_total), showarrow:false, font:{size:16}}]}, CFG);

  const gs = [...D.syms].sort((a,b)=>a.gain_pct-b.gain_pct);
  Plotly.newPlot("figGain", [{
    type:"bar", orientation:"h", y:gs.map(s=>s.sym), x:gs.map(s=>s.gain_pct),
    marker:{color:gs.map(s=>s.gain_pct>=0?"#2f7d57":"#b5442a")},
    text:gs.map(s=>fmtPct(s.gain_pct)), textposition:"outside",
    hovertemplate:"%{y}: %{x:+.1f}%<extra></extra>",
  }], {...LAY, title:{text:"各标的未实现盈亏率"}, xaxis:{ticksuffix:"%"}}, CFG);
}
document.getElementById("notes").innerHTML =
  "说明：VINIX/VIEIX 为 401(k) 机构信托，采用表格快照价（公募基金同名代码的 NAV 不适用）；" +
  "M1 账户头寸的股数按 快照市值/现价 反推；已归属 RSU 的成本记为归属后市值（盈亏记 0）；" +
  "「当日变动」= 股数 ×（最新收盘 − 前一收盘）。数据：Yahoo Finance。";
</script>
</body>
</html>
"""


def main():
    rows, asof = gather()
    d = build(rows, asof)
    d["rows_cash"] = [r for r in rows if r["cash"]]
    cash_total = d["cash_total"]
    d["perf"] = compute_performance(rows, cash_total)
    html = HTML.replace("%%PLOTLY%%", _js("plotly.min.js"))
    html = html.replace("%%DATA%%", json.dumps(d, ensure_ascii=False, default=str))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML written: {OUT}")
    print(f"total ${d['grand_total']:,.0f} | gain {d['gain_total']:+,.0f} "
          f"({d['gain_pct']:+.1f}%) | day {d['day_total']:+,.0f}")
    for p in d["perf"]["periods"]:
        print(f"  {p['label']:>4}: {p['pct']:+.1f}%  ({p['usd']:+,.0f})")
    pm, b = d["perf"]["port_m"], d["perf"]["bench"]
    print(f"  port: cum {pm['cum']:+.1f}% vol {pm['vol']:.1f}% sharpe {pm['sharpe']:.2f} "
          f"maxDD {pm['max_dd']:.1f}%")
    for k in ["SPY", "QQQ", "SMH"]:
        print(f"  {k:>4}: cum {b[k]['m']['cum']:+.1f}% vol {b[k]['m']['vol']:.1f}% "
              f"sharpe {b[k]['m']['sharpe']:.2f} | beta {b[k]['rel']['beta']:.2f} "
              f"alpha {b[k]['rel']['alpha']:+.1f}% IR {b[k]['rel']['ir']:.2f}")


if __name__ == "__main__":
    main()
