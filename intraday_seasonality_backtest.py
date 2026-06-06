"""
intraday_seasonality_backtest.py  (v3)
======================================
Backtest an intraday same-slot seasonality stock-selection table, inspired by
HKS 2010, and separate momentum-in-disguise from genuine cross-sectional
seasonality with four diagnostic gates:

  [1] Walk-forward OOS: each day uses only the previous LOOKBACK days.
  [2] Pick concentration: top-name share >~30% is a momentum warning.
  [3] Two-way demean isolation: remove stock-level and slot-level effects,
      leaving only the stock x slot interaction.
  [4] Regime split + random null: high-vol-only edge suggests momentum;
      failure to beat random suggests no stock-selection information.

Usage: pip install yfinance pandas numpy ; python intraday_seasonality_backtest.py
Data: yfinance 30m only reaches roughly 60 days. Swap load_bars() for longer history.
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
    """Read a local fallback value from git-ignored config_local.py."""
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
    """Small Polygon pagination helper. Free tiers may rate-limit; retry on 429."""
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
    """Keep only dates where every ticker has all 13 regular-session slots."""
    ok=(lr.groupby(["ticker","date"])["slot"].nunique()
          .unstack("ticker")
          .eq(expected_slots)
          .all(axis=1))
    return lr[lr["date"].isin(ok[ok].index)].copy()

def _pivot(lr): return lr.pivot_table(index=["date","slot"],columns="ticker",values="ret")

def _two_way_demean(piv):
    """Daily two-way demean: remove slot and stock means, leaving stock x slot interaction."""
    out=piv.copy().astype(float)
    for d,b in piv.groupby(level="date"):
        M=b.values.astype(float)
        out.loc[b.index]=M-np.nanmean(M,1,keepdims=True)-np.nanmean(M,0,keepdims=True)+np.nanmean(M)
    return out

def backtest(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS, mode="raw", select="argmax", seed=0):
    """mode='raw' trades raw returns; mode='season' trades two-way-demeaned returns."""
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
    """Dollar-neutral variant: long same-slot argmax and short same-slot argmin.

    Net = (r_long - r_short) - 2 * round-trip cost, one cost per leg.
    """
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
    """Daily absolute market-move proxy, matching regime_split."""
    return _pivot(lr).groupby(level="date").mean().mean(axis=1).abs()

def daily_net_bps(res):
    """Aggregate per-trade net into daily bps; the day is the cluster-aware unit."""
    return res.groupby("date")["net"].sum() * 1e4

def bootstrap_daily(daily, n=10000, seed=0):
    """Bootstrap days, not trades, to estimate a 95% CI for daily mean net bps."""
    rng = np.random.default_rng(seed); d = daily.values
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def drop_top_vol_days(res, lr, ks=(0, 1, 3, 5)):
    """Drop top-K high-volatility days, sorted by ex-ante market move, then recompute mean/t."""
    daily = daily_net_bps(res); vol = _market_move(lr).reindex(daily.index)
    order = vol.sort_values(ascending=False).index; out = {}
    for k in ks:
        keep = daily.drop(order[:k])
        t = keep.mean() / (keep.std(ddof=1) / np.sqrt(len(keep)) + 1e-12)
        out[k] = (float(keep.mean()), float(t), int(len(keep)))
    return out

def season_excess_by_regime(lr, lookback=LOOKBACK, seeds=20):
    """Compute seasonality excess over random separately on high/low volatility days."""
    sa = backtest(lr, lookback=lookback, mode="season", select="argmax")
    rs = [backtest(lr, lookback=lookback, mode="season", select="random", seed=k) for k in range(seeds)]
    vol = _market_move(lr); med = vol.median()
    def buck(df, hi):
        m = df["date"].map(lambda d: (vol.get(d, 0) >= med) == hi)
        return df[m]["gross"].mean() * 1e4
    return {nm: float(buck(sa, hi) - np.mean([buck(r, hi) for r in rs]))
            for hi, nm in [(True, "high"), (False, "low")]}

def attribution(res):
    """Break backtest P&L down by ticker and slot. res comes from backtest(mode='raw')."""
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
    print("By ticker (top 5, sorted by total net):"); print(by_t.head(5).round(1).to_string())
    print(f"\nTop ticker contributes {share:.0f}% of total net P&L  (>~70% => single-name momentum, not broad seasonality)")
    print("\nBy slot (mean net bps):"); print(by_s["mean_net_bps"].round(2).to_string())

def _strategy_stats(res, lr):
    hi,lo=regime_split(lr,res)
    return dict(net_bps=float(res["net"].mean()*1e4),
                win_pct=float((res["net"]>0).mean()*100),
                cum_pct=float(((1+res["net"]).cumprod().iloc[-1]-1)*100),
                hi_bps=float(hi),
                lo_bps=float(lo))

def compare_long_only_vs_ls(lr):
    """Compare original long-only and long/short neutral versions."""
    lo=backtest(lr,mode="raw")
    ls=backtest_ls(lr)
    rows=pd.DataFrame({"long_only":_strategy_stats(lo,lr),
                       "long_short":_strategy_stats(ls,lr)}).T
    print("long-only vs long/short (net bps/trade, win, cumulative, regime):")
    print(rows.rename(columns=dict(net_bps="net/trade bps",win_pct="win %",
                                   cum_pct="cumulative net %",hi_bps="high-vol bps",
                                   lo_bps="low-vol bps")).round(2).to_string())
    return rows

def regime_split(lr,res):
    mkt=_market_move(lr); med=mkt.median()
    hi=res[res["date"].map(lambda d:mkt.get(d,0)>=med)]["net"].mean()*1e4
    lo=res[res["date"].map(lambda d:mkt.get(d,0)< med)]["net"].mean()*1e4
    return hi,lo

def evaluate(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS, random_seeds=20):
    """Return diagnostic metrics; diagnostics() only prints this payload."""
    raw=backtest(lr,lookback=lookback,cost_rt_bps=cost_rt_bps,mode="raw")
    if raw.empty:
        raise ValueError("no trades produced; check input panel")
    g,n=raw["gross"]*1e4, raw["net"]*1e4
    s_arg=backtest(lr,lookback=lookback,mode="season",select="argmax")["gross"].mean()*1e4   # Isolated seasonality signal.
    s_rng=float(np.mean([backtest(lr,lookback=lookback,mode="season",select="random",seed=k)["gross"].mean()
                         for k in range(random_seeds)])*1e4)                                  # Random-selection floor.
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
        print("No trades produced; check input data"); return
    print(f"Trades {m['n_trades']} | lookback {LOOKBACK}d | cost {COST_RT_BPS}bps\n")
    print(f"[1] Raw long-only     gross {m['raw_gross_bps']:+.2f} / net {m['raw_net_bps']:+.2f} bps | win {m['win_rate_pct']:.1f}%"
          f" | cumulative net {m['cumulative_net_pct']:+.1f}%")
    print(f"[2] Pick concentration   top name {m['concentration_pct']:.0f}%   ({'>30%=momentum contamination warning' if m['concentration_pct']>30 else 'acceptable'})")
    print(f"[3] Isolated seasonality  argmax {m['season_argmax_bps']:+.2f}bps  vs  random floor {m['season_random_bps']:+.2f}bps"
          f"  -> excess {m['season_excess_bps']:+.2f}bps")
    print(f"[4] Regime split (net)  high-vol {m['regime_hi_bps']:+.2f} / low-vol {m['regime_lo_bps']:+.2f} bps")
    print("\nVerdict:")
    # Argmax has a ~1.5-2bps small-universe selection-bias floor, so the threshold keeps margin.
    excess = m["season_excess_bps"]
    real_season = excess > 3.0
    print(f"  - real seasonality: {'yes (isolated +%.1fbps is well above the ~1.5-2bps selection floor)'%excess if real_season else 'no (isolated +%.1fbps is near the selection floor, consistent with trend/momentum noise)'%excess}")
    print(f"  - tradability: raw net/trade {'>0' if m['raw_net_bps']>0 else '<=0 after cost'}"
          f" ; breakeven round-trip cost {m['raw_gross_bps']:.2f}bps")
    print(f"  - momentum warning: concentration {m['concentration_pct']:.0f}% , high-vol days {'dominate returns' if m['regime_hi_bps']>2*max(m['regime_lo_bps'],0.01) else 'do not dominate'}")

def deep_diagnostics(lr):
    m=evaluate(lr)
    drops=" | ".join(f"k={k} {v[0]:+.2f}(t={v[1]:+.2f},n={v[2]})"
                     for k,v in m["drop_top_vol_days"].items())
    print(f"Effective sample: {m['n_test_days']} trading days ({m['n_trades']} trades = {m['n_test_days']} days x 13 correlated slots)")
    print(f"Daily net: {m['daily_net_mean_bps']:+.2f} bps | daily bootstrap 95% CI "
          f"[{m['daily_boot_lo_bps']:+.2f}, {m['daily_boot_hi_bps']:+.2f}]")
    print(f"Drop extreme days: {drops}   (collapse after dropping them => few-day regime artifact)")
    print(f"Isolated seasonality by regime: high {m['season_excess_high_bps']:+.2f} / "
          f"low {m['season_excess_low_bps']:+.2f}   (only high positive => regime/momentum, not stable seasonality)")
    print()
    print_attribution(backtest(lr,mode="raw"))

def _summary_line(label, m):
    return (f"== {label} ==   raw_net {m['raw_net_bps']:+.2f} | "
            f"season_excess {m['season_excess_bps']:+.2f} | conc {m['concentration_pct']:.0f}% | "
            f"regime hi/lo {m['regime_hi_bps']:+.2f}/{m['regime_lo_bps']:+.2f} | "
            f"days {m['n_test_days']}")

def period_evaluation(lr, freq="Q", lookback=LOOKBACK):
    """Split a long sample by calendar period and run evaluate on each segment."""
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
    """Polygon long-window report: full window plus calendar sub-samples."""
    if end is None:
        end=pd.Timestamp.now(tz=TZ).date().isoformat()
    if start is None:
        start=(pd.Timestamp(end)-pd.DateOffset(months=18)).date().isoformat()
    bars=load_bars_polygon(tickers,start,end,api_key=api_key)
    lr=keep_full_sessions(to_slot_returns(bars))
    print(f"Data window {start} -> {end} | full sessions {lr['date'].nunique()} | tickers {len(tickers)}")
    print(_summary_line("full 18mo",evaluate(lr)))
    for label,m in period_evaluation(lr,freq=freq).items():
        print(_summary_line(label,m))
    return lr

if __name__=="__main__":
    bars=load_bars(UNIVERSE); lr=to_slot_returns(bars); diagnostics(lr)
