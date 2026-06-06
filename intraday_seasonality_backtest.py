"""
intraday_seasonality_backtest.py  (v3)
======================================
回测「日内同槽周期性」选股表(HKS 2010 的散户实现),并用四道关把
"靠捡趋势赚钱(动量假象)" 和 "真·横截面 seasonality edge" 彻底分开:

  [1] Walk-forward 出样本 : 每天只用之前 LOOKBACK 天选表、在未见过的当天交易(默认即是)。
  [2] 选股集中度          : 头名占比 >~30% 警惕——真 seasonality 应是不同票主导不同槽。
  [3] 双向去均值隔离       : 去掉「个股水平」与「时段水平」,只留 stock×slot 交互。
                          raw 赚钱但此项掉回随机地板 => 是趋势/动量,不是 seasonality。
  [4] 行情分桶 + 随机零假设: edge 只在高波动日出现=动量;不显著高于随机=选股没信息。

用法: pip install yfinance pandas numpy ; python intraday_seasonality_backtest.py
数据: yfinance 30m 仅 ~60 天。换更长历史只需替换 load_bars()。
"""
import numpy as np, pandas as pd
import os, time, json
from urllib.parse import urlencode, urlparse, parse_qsl, urlunparse
from urllib.error import HTTPError
from urllib.request import urlopen

UNIVERSE    = ["AAPL","ABNB","ORCL","AMD","JPM","AMZN","ASML","XOM","AVGO"]
INTERVAL, PERIOD = "30m", "60d"
LOOKBACK    = 30
COST_RT_BPS = 3.0
TZ          = "America/New_York"

def _local_config_value(name, default=None):
    """从 git-ignored config_local.py 兜底读取本机配置。"""
    try:
        import config_local as cfg
        return getattr(cfg, name, default)
    except Exception:
        return default

def load_bars(tickers, interval=INTERVAL, period=PERIOD):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False)
    return {t:(raw[t] if len(tickers)>1 else raw)[["Open","Close"]].dropna() for t in tickers}

def _add_query(url, **params):
    p=urlparse(url); q=dict(parse_qsl(p.query)); q.update({k:v for k,v in params.items() if v is not None})
    return urlunparse(p._replace(query=urlencode(q)))

def _polygon_get(url, api_key, sleep=12.5):
    """Polygon 分页请求小工具。免费档可能限速,遇到 429 稍等后重试。"""
    url=_add_query(url, apiKey=api_key)
    for i in range(5):
        try:
            with urlopen(url, timeout=30) as r:
                data=json.loads(r.read().decode("utf-8"))
            time.sleep(sleep)
            return data
        except HTTPError as e:
            if e.code == 429 and i<4:
                time.sleep(65)
                continue
            raise
        except Exception as e:
            if "HTTP Error 429" in str(e) and i<4:
                time.sleep(65)
                continue
            raise

def load_bars_polygon(tickers, start, end, api_key=None):
    """30m bars from Polygon. start/end = 'YYYY-MM-DD'. Returns the same dict shape as load_bars().
    api_key from arg or env POLYGON_API_KEY. Use ADJUSTED prices. Return tz-aware ET index."""
    api_key=api_key or os.environ.get("POLYGON_API_KEY") or _local_config_value("POLYGON_API_KEY")
    if not api_key:
        raise ValueError("Polygon API key required: pass api_key, set POLYGON_API_KEY, or save config_local.py")
    out={}
    for t in tickers:
        url=(f"https://api.polygon.io/v2/aggs/ticker/{t}/range/30/minute/{start}/{end}"
             f"?adjusted=true&sort=asc&limit=50000")
        rows=[]
        while url:
            data=_polygon_get(url,api_key)
            rows.extend(data.get("results",[]))
            url=data.get("next_url")
        if not rows:
            out[t]=pd.DataFrame(columns=["Open","Close"],
                                index=pd.DatetimeIndex([],tz=TZ))
            continue
        df=pd.DataFrame(rows)
        idx=pd.to_datetime(df["t"],unit="ms",utc=True).dt.tz_convert(TZ)
        bars=(pd.DataFrame({"Open":df["o"].astype(float).values,
                            "Close":df["c"].astype(float).values},
                           index=pd.DatetimeIndex(idx))
                .between_time("09:30","15:30")
                .dropna())
        out[t]=bars
    return out

def to_slot_returns(bars):
    fr=[]
    for t,df in bars.items():
        df=df.copy(); idx=df.index.tz_convert(TZ) if df.index.tz is not None else df.index
        d=pd.Series(idx.date,index=df.index)
        df=df.assign(date=d.values,slot=df.groupby(d.values).cumcount(),
                     ret=df["Close"]/df["Open"]-1.0,ticker=t)
        fr.append(df[["date","slot","ticker","ret"]])
    return pd.concat(fr,ignore_index=True)

def keep_full_sessions(lr, expected_slots=13):
    """只保留每票每天都有完整 13 个 regular-session 槽的日期,避免早收盘/缺 bar 污染槽编号。"""
    ok=(lr.groupby(["ticker","date"])["slot"].nunique()
          .unstack("ticker")
          .eq(expected_slots)
          .all(axis=1))
    return lr[lr["date"].isin(ok[ok].index)].copy()

def _pivot(lr): return lr.pivot_table(index=["date","slot"],columns="ticker",values="ret")

def _two_way_demean(piv):
    """逐日去掉 行(时段)均值 与 列(个股)均值,只留 stock×slot 交互 -> 隔离 seasonality。"""
    out=piv.copy().astype(float)
    for d,b in piv.groupby(level="date"):
        M=b.values.astype(float)
        out.loc[b.index]=M-np.nanmean(M,1,keepdims=True)-np.nanmean(M,0,keepdims=True)+np.nanmean(M)
    return out

def backtest(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS, mode="raw", select="argmax", seed=0):
    """mode: 'raw'=原版长仓; 'season'=双向去均值后(隔离seasonality)。select: 'argmax'|'random'。"""
    piv=_pivot(lr); work=_two_way_demean(piv) if mode=="season" else piv
    dates=sorted(lr["date"].unique()); slots=sorted(lr["slot"].unique())
    cols=list(work.columns); dlev=work.index.get_level_values("date")
    rng=np.random.default_rng(seed); rows=[]
    for i in range(lookback,len(dates)):
        d=dates[i]; sub=work[dlev.isin(set(dates[i-lookback:i]))]
        for s in slots:
            tr=sub.xs(s,level="slot")
            if tr.empty: continue
            sc=tr.mean().dropna()
            if sc.empty: continue
            pick=sc.idxmax() if select=="argmax" else cols[rng.integers(len(cols))]
            try: rz=work.loc[(d,s),pick]
            except KeyError: continue
            if pd.isna(rz): continue
            rows.append(dict(date=d,slot=s,pick=pick,gross=rz,net=rz-cost_rt_bps/1e4))
    return pd.DataFrame(rows)

def backtest_ls(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS):
    """每个 (day, slot): 用前 lookback 天同槽均值排序,long 头名 / short 尾名(美元中性)。
    净 = (r_long - r_short) - 2*往返成本(两条腿各付一次)。不改动 backtest 本体。"""
    piv = _pivot(lr)
    dates = sorted(lr["date"].unique()); slots = sorted(lr["slot"].unique())
    dlev = piv.index.get_level_values("date"); rows = []
    for i in range(lookback, len(dates)):
        d = dates[i]; sub = piv[dlev.isin(set(dates[i-lookback:i]))]
        for s in slots:
            tr = sub.xs(s, level="slot"); sc = tr.mean().dropna()
            if len(sc) < 2:
                continue
            L, S = sc.idxmax(), sc.idxmin()
            try:
                rl = piv.loc[(d, s), L]; rs = piv.loc[(d, s), S]
            except KeyError:
                continue
            if pd.isna(rl) or pd.isna(rs):
                continue
            rows.append(dict(date=d, slot=s, long=L, short=S,
                             gross=float(rl - rs), net=float((rl - rs) - 2 * cost_rt_bps / 1e4)))
    return pd.DataFrame(rows)

def _market_move(lr):
    """当日 |大盘| 代理(与 regime_split 用的口径一致)。"""
    return _pivot(lr).groupby(level="date").mean().mean(axis=1).abs()

def daily_net_bps(res):
    """把逐笔 net 聚合成日收益序列:每天 = 当天 13 槽 net 之和(bps)。
    这是 cluster-aware 推断的单位——同一天 13 槽不是独立观测。"""
    return res.groupby("date")["net"].sum() * 1e4

def bootstrap_daily(daily, n=10000, seed=0):
    """对'天'重采样(而非对'笔'),得到日均净收益的 95% CI。返回 (mean, lo, hi) bps。"""
    rng = np.random.default_rng(seed); d = daily.values
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def drop_top_vol_days(res, lr, ks=(0, 1, 3, 5)):
    """按【事前】|大盘|从高到低排序,逐步抽掉 top-K 高波动日后重算日均净收益+t值。
    注意:按市场波动(事前)排序,不按收益(避免用结果挑日子)。
    返回 {k: (mean_bps, t_stat, n_days)}。edge 一抽就塌 => 几天 regime 行情。"""
    daily = daily_net_bps(res); vol = _market_move(lr).reindex(daily.index)
    order = vol.sort_values(ascending=False).index; out = {}
    for k in ks:
        keep = daily.drop(order[:k])
        t = keep.mean() / (keep.std(ddof=1) / np.sqrt(len(keep)) + 1e-12)
        out[k] = (float(keep.mean()), float(t), int(len(keep)))
    return out

def season_excess_by_regime(lr, lookback=LOOKBACK, seeds=20):
    """对【隔离 seasonality 项】(双向去均值 argmax) 也按高/低波动日分桶,
    各桶内算 argmax 相对随机地板的超出。返回 {'high':excess_bps, 'low':excess_bps}。
    真 seasonality 应两桶都为正;若只在高波动桶为正 => 那点'seasonality'本身是 regime/动量的。"""
    sa = backtest(lr, lookback=lookback, mode="season", select="argmax")
    rs = [backtest(lr, lookback=lookback, mode="season", select="random", seed=k) for k in range(seeds)]
    vol = _market_move(lr); med = vol.median()
    def buck(df, hi):
        m = df["date"].map(lambda d: (vol.get(d, 0) >= med) == hi)
        return df[m]["gross"].mean() * 1e4
    return {nm: float(buck(sa, hi) - np.mean([buck(r, hi) for r in rs]))
            for hi, nm in [(True, "high"), (False, "low")]}

def attribution(res):
    """把回测结果按【票】和【槽】拆解 P&L。res 来自 backtest(mode='raw')。"""
    by_ticker = (res.groupby("pick")
                    .agg(trades=("net", "size"),
                         total_net_bps=("net", lambda x: float(x.sum() * 1e4)),
                         mean_net_bps =("net", lambda x: float(x.mean() * 1e4)))
                    .sort_values("total_net_bps", ascending=False))
    by_slot   = (res.groupby("slot")
                    .agg(trades=("net", "size"),
                         mean_net_bps=("net", lambda x: float(x.mean() * 1e4)))
                    .sort_index())
    total = res["net"].sum() * 1e4
    top_ticker_share_pct = float(by_ticker["total_net_bps"].iloc[0] / total * 100) if total else float("nan")
    return by_ticker, by_slot, top_ticker_share_pct

def print_attribution(res):
    by_t, by_s, share = attribution(res)
    print("按票 (前5, 按总净排序):"); print(by_t.head(5).round(1).to_string())
    print(f"\n头名贡献占总净 P&L 的 {share:.0f}%  (>~70% => 单票主导=动量, 非广义seasonality)")
    print("\n按槽 (均净 bps):"); print(by_s["mean_net_bps"].round(2).to_string())

def _strategy_stats(res, lr):
    hi,lo=regime_split(lr,res)
    return dict(net_bps=float(res["net"].mean()*1e4),
                win_pct=float((res["net"]>0).mean()*100),
                cum_pct=float(((1+res["net"]).cumprod().iloc[-1]-1)*100),
                hi_bps=float(hi),
                lo_bps=float(lo))

def compare_long_only_vs_ls(lr):
    """对比原版 long-only 与 long/short 中性版:收益、胜率、累计、regime split。"""
    lo=backtest(lr,mode="raw")
    ls=backtest_ls(lr)
    rows=pd.DataFrame({"long_only":_strategy_stats(lo,lr),
                       "long_short":_strategy_stats(ls,lr)}).T
    print("long-only vs long/short (net bps/trade, win, cumulative, regime):")
    print(rows.rename(columns=dict(net_bps="净/笔bps",win_pct="胜率%",
                                   cum_pct="累计净%",hi_bps="高波动bps",
                                   lo_bps="低波动bps")).round(2).to_string())
    return rows

def regime_split(lr,res):
    mkt=_market_move(lr); med=mkt.median()
    hi=res[res["date"].map(lambda d:mkt.get(d,0)>=med)]["net"].mean()*1e4
    lo=res[res["date"].map(lambda d:mkt.get(d,0)< med)]["net"].mean()*1e4
    return hi,lo

def evaluate(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS, random_seeds=20):
    """返回判别指标(diagnostics 只负责打印它)。不要改动任何计算口径。"""
    raw=backtest(lr,lookback=lookback,cost_rt_bps=cost_rt_bps,mode="raw")
    if raw.empty:
        raise ValueError("no trades produced; check input panel")
    g,n=raw["gross"]*1e4, raw["net"]*1e4
    s_arg=backtest(lr,lookback=lookback,mode="season",select="argmax")["gross"].mean()*1e4   # 真seasonality信号
    s_rng=float(np.mean([backtest(lr,lookback=lookback,mode="season",select="random",seed=k)["gross"].mean()
                         for k in range(random_seeds)])*1e4)                                  # 随机地板
    hi,lo=regime_split(lr,raw)
    daily=daily_net_bps(raw)
    daily_mean,daily_lo,daily_hi=bootstrap_daily(daily)
    sxr=season_excess_by_regime(lr,lookback=lookback,seeds=random_seeds)
    return dict(
        n_trades=int(len(raw)),
        n_test_days=int(daily.size),
        raw_gross_bps=float(g.mean()),
        raw_net_bps=float(n.mean()),
        win_rate_pct=float((n>0).mean()*100),
        cumulative_net_pct=float(((1+raw["net"]).cumprod().iloc[-1]-1)*100),
        concentration_pct=float(raw["pick"].value_counts(normalize=True).iloc[0]*100),
        season_argmax_bps=float(s_arg),
        season_random_bps=float(s_rng),
        season_excess_bps=float(s_arg-s_rng),
        regime_hi_bps=float(hi),
        regime_lo_bps=float(lo),
        daily_net_mean_bps=float(daily_mean),
        daily_boot_lo_bps=float(daily_lo),
        daily_boot_hi_bps=float(daily_hi),
        drop_top_vol_days=drop_top_vol_days(raw,lr),
        season_excess_high_bps=float(sxr["high"]),
        season_excess_low_bps=float(sxr["low"]),
    )

def diagnostics(lr):
    try:
        m=evaluate(lr)
    except ValueError:
        print("无交易,检查数据"); return
    print(f"交易笔数 {m['n_trades']} | 回看 {LOOKBACK}d | 成本 {COST_RT_BPS}bps\n")
    print(f"[1] 原版长仓     毛 {m['raw_gross_bps']:+.2f} / 净 {m['raw_net_bps']:+.2f} bps | 胜率 {m['win_rate_pct']:.1f}%"
          f" | 累计净 {m['cumulative_net_pct']:+.1f}%")
    print(f"[2] 选股集中度   头名占 {m['concentration_pct']:.0f}%   ({'>30%=动量污染嫌疑' if m['concentration_pct']>30 else '尚可'})")
    print(f"[3] 隔离seasonality  argmax {m['season_argmax_bps']:+.2f}bps  vs  随机地板 {m['season_random_bps']:+.2f}bps"
          f"  -> 超出 {m['season_excess_bps']:+.2f}bps")
    print(f"[4] 行情分桶(净)  高波动日 {m['regime_hi_bps']:+.2f} / 低波动日 {m['regime_lo_bps']:+.2f} bps")
    print("\n判别:")
    # 注:argmax 在小universe上本身有 ~1.5-2bps 选择偏差地板,故阈值取 ~3bps 留余量
    excess = m["season_excess_bps"]
    real_season = excess > 3.0
    print(f"  · seasonality 是否真实: {'是(隔离后 +%.1fbps 远超选择偏差地板~1.5-2bps)'%excess if real_season else '否(隔离后仅 +%.1fbps ≈ 选择偏差地板 = 趋势/动量假象)'%excess}")
    print(f"  · 可交易性: 原版净/笔 {'>0' if m['raw_net_bps']>0 else '<=0 (扣成本亏损)'}"
          f" ; 盈亏平衡往返成本 {m['raw_gross_bps']:.2f}bps")
    print(f"  · 动量嫌疑: 集中度 {m['concentration_pct']:.0f}% , 高波动日{'独占收益' if m['regime_hi_bps']>2*max(m['regime_lo_bps'],0.01) else '未独占'}")

def deep_diagnostics(lr):
    m=evaluate(lr)
    drops=" | ".join(f"k={k} {v[0]:+.2f}(t={v[1]:+.2f},n={v[2]})"
                     for k,v in m["drop_top_vol_days"].items())
    print(f"有效样本: {m['n_test_days']} 个可交易日 ({m['n_trades']} 笔 = {m['n_test_days']} 天 × 13 相关槽)")
    print(f"日均净: {m['daily_net_mean_bps']:+.2f} bps | 按天 bootstrap 95% CI "
          f"[{m['daily_boot_lo_bps']:+.2f}, {m['daily_boot_hi_bps']:+.2f}]")
    print(f"抽极端日: {drops}   (一抽就塌 => 几天行情)")
    print(f"隔离seasonality 分regime: high {m['season_excess_high_bps']:+.2f} / "
          f"low {m['season_excess_low_bps']:+.2f}   (只 high 为正 => regime/动量, 非稳定seasonality)")
    print()
    print_attribution(backtest(lr,mode="raw"))

def _summary_line(label, m):
    return (f"== {label} ==   raw_net {m['raw_net_bps']:+.2f} | "
            f"season_excess {m['season_excess_bps']:+.2f} | conc {m['concentration_pct']:.0f}% | "
            f"regime hi/lo {m['regime_hi_bps']:+.2f}/{m['regime_lo_bps']:+.2f} | "
            f"days {m['n_test_days']}")

def period_evaluation(lr, freq="Q", lookback=LOOKBACK):
    """按日历区间拆分长样本,逐段跑同一 evaluate 口径,看 edge 是否跨 regime 稳定。"""
    out={}
    dts=pd.to_datetime(lr["date"])
    for p in sorted(dts.dt.to_period(freq).unique()):
        sub=lr[dts.dt.to_period(freq).eq(p)]
        try:
            out[str(p)]=evaluate(sub,lookback=lookback)
        except ValueError:
            continue
    return out

def long_window_report_polygon(tickers=UNIVERSE, start=None, end=None, api_key=None, freq="Q"):
    """Polygon 长窗口实盘数据报告:full window + 分季度子样本。"""
    if end is None:
        end=pd.Timestamp.now(tz=TZ).date().isoformat()
    if start is None:
        start=(pd.Timestamp(end)-pd.DateOffset(months=18)).date().isoformat()
    bars=load_bars_polygon(tickers,start,end,api_key=api_key)
    lr=keep_full_sessions(to_slot_returns(bars))
    print(f"数据窗口 {start} -> {end} | 完整交易日 {lr['date'].nunique()} | tickers {len(tickers)}")
    print(_summary_line("full 18mo",evaluate(lr)))
    for label,m in period_evaluation(lr,freq=freq).items():
        print(_summary_line(label,m))
    return lr

if __name__=="__main__":
    bars=load_bars(UNIVERSE); lr=to_slot_returns(bars); diagnostics(lr)
