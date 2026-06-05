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

UNIVERSE    = ["AAPL","ABNB","ORCL","AMD","JPM","AMZN","ASML","XOM","AVGO"]
INTERVAL, PERIOD = "30m", "60d"
LOOKBACK    = 30
COST_RT_BPS = 3.0
TZ          = "America/New_York"

def load_bars(tickers, interval=INTERVAL, period=PERIOD):
    import yfinance as yf
    raw = yf.download(tickers, period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False)
    return {t:(raw[t] if len(tickers)>1 else raw)[["Open","Close"]].dropna() for t in tickers}

def to_slot_returns(bars):
    fr=[]
    for t,df in bars.items():
        df=df.copy(); idx=df.index.tz_convert(TZ) if df.index.tz is not None else df.index
        d=pd.Series(idx.date,index=df.index)
        df=df.assign(date=d.values,slot=df.groupby(d.values).cumcount(),
                     ret=df["Close"]/df["Open"]-1.0,ticker=t)
        fr.append(df[["date","slot","ticker","ret"]])
    return pd.concat(fr,ignore_index=True)

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

def regime_split(lr,res):
    mkt=_pivot(lr).groupby(level="date").mean().mean(axis=1).abs(); med=mkt.median()
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
    return dict(
        n_trades=int(len(raw)),
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

if __name__=="__main__":
    bars=load_bars(UNIVERSE); lr=to_slot_returns(bars); diagnostics(lr)
