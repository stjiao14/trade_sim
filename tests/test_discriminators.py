"""合成对照回归关:锁住 '无edge / 真seasonality / 动量陷阱' 三种判读。
任何改动后必须全过,否则说明判别逻辑被动坏了。"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np, pandas as pd
import intraday_seasonality_backtest as bt

N_TICKERS, LOOKBACK, COST_RT_BPS = 9, 30, 3.0

def make_panel(seed, season=0.0, trend=0.0, n=N_TICKERS, days=60):
    """合成半小时收益面板。season=真·stock×slot edge;trend=每票持续drift。
    rng 抽取顺序固定 -> 结果确定可复现。"""
    rng = np.random.default_rng(seed)
    tickers = [f"T{i}" for i in range(n)]
    dts = pd.bdate_range("2025-01-02", periods=days, tz="America/New_York")
    mu = rng.normal(0, season, (n, 13))
    dr = rng.normal(0, trend, n)
    frames = []
    for ti, t in enumerate(tickers):
        ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30*k)
                               for d in dts for k in range(13)])
        slot = np.tile(np.arange(13), len(dts))
        ret = mu[ti, slot] + dr[ti] + rng.normal(0, 0.004, len(ts))
        bars = pd.DataFrame({"Open": 100.0, "Close": 100*(1+ret)}, index=ts)
        frames.append(bt.to_slot_returns({t: bars}))
    return pd.concat(frames, ignore_index=True)

def _eval(**kw):
    return bt.evaluate(make_panel(42, **kw), lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS)

def test_pure_noise_has_no_edge():
    m = _eval(season=0.0, trend=0.0)
    assert m["season_excess_bps"] < 3.0,  m   # 不是真seasonality(≈选择偏差地板)
    assert m["raw_net_bps"]      < 0.0,  m    # 扣成本不可交易
    assert m["concentration_pct"] < 30.0, m   # 无动量集中

def test_true_seasonality_detected():
    m = _eval(season=0.001, trend=0.0)
    assert m["season_excess_bps"] > 4.0,  m   # 隔离后远超地板 => 真信号
    assert m["concentration_pct"] < 30.0, m   # 不同票分散在不同槽

def test_trend_trap_is_flagged_as_momentum():
    m = _eval(season=0.0, trend=0.0008)
    assert m["raw_net_bps"]      > 0.0,  m    # 看起来能赚钱(陷阱)
    assert m["season_excess_bps"] < 3.0, m    # ...但seasonality是假象
    assert m["concentration_pct"] > 35.0, m   # 动量污染被亮红灯

def make_true_uniform(seed=42, season=0.001, n=9, days=60):
    """均匀的真 seasonality(非 regime 驱动)。"""
    rng = np.random.default_rng(seed); tk = [f"T{i}" for i in range(n)]
    dts = pd.bdate_range("2025-01-02", periods=days, tz="America/New_York")
    mu = rng.normal(0, season, (n, 13)); fr = []
    for ti, t in enumerate(tk):
        ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30*k)
                               for d in dts for k in range(13)])
        slot = np.tile(np.arange(13), len(dts)); ret = mu[ti, slot] + rng.normal(0, 0.004, len(ts))
        fr.append(bt.to_slot_returns({t: pd.DataFrame({"Open": 100., "Close": 100*(1+ret)}, index=ts)}))
    return pd.concat(fr, ignore_index=True)

def make_regime_trap(seed=7, n=9, days=60, n_vol=5):
    """纯 regime 动量陷阱:每票持续相对强弱(drift)只在少数高波动日兑现;平日纯噪声。"""
    rng = np.random.default_rng(seed); tk = [f"T{i}" for i in range(n)]
    dts = pd.bdate_range("2025-01-02", periods=days, tz="America/New_York")
    dr = rng.normal(0, 0.004, n)
    volday = np.zeros(days, bool); volday[rng.choice(days, n_vol, replace=False)] = True; fr = []
    for ti, t in enumerate(tk):
        ts = pd.DatetimeIndex([d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30*k)
                               for d in dts for k in range(13)])
        slot = np.tile(np.arange(13), len(dts)); dayof = np.repeat(np.arange(days), 13)
        ret = np.where(volday[dayof], dr[ti], 0.0) + rng.normal(0, 0.004, len(ts)) * np.where(volday[dayof], 3., 1.)
        fr.append(bt.to_slot_returns({t: pd.DataFrame({"Open": 100., "Close": 100*(1+ret)}, index=ts)}))
    return pd.concat(fr, ignore_index=True)

def test_genuine_seasonality_is_regime_robust():
    lr = make_true_uniform()
    res = bt.backtest(lr, mode="raw")
    dvd = bt.drop_top_vol_days(res, lr)
    reg = bt.season_excess_by_regime(lr)
    # 抽掉 top-5 高波动日,edge 仍显著为正(真信号不靠那几天)
    mean5, t5, _ = dvd[5]
    assert mean5 > 0 and t5 > 1.0, dvd
    # 隔离 seasonality 在高、低波动两个 regime 都为正
    assert reg["high"] > 2.0 and reg["low"] > 2.0, reg

def test_regime_momentum_trap_collapses():
    lr = make_regime_trap()
    res = bt.backtest(lr, mode="raw")
    dvd = bt.drop_top_vol_days(res, lr)
    reg = bt.season_excess_by_regime(lr)
    # 抽掉高波动日后不再显著为正(edge 靠那几天)
    _, t5, _ = dvd[5]
    assert t5 < 1.0, dvd
    # 隔离 seasonality 不是两个 regime 都为真(至少一个 <=地板~1.5/为负)
    assert not (reg["high"] > 2.0 and reg["low"] > 2.0), reg

def test_attribution_flags_single_name_dominance():
    # trend trap: 单票应主导 P&L
    trap = bt.backtest(make_panel(42, trend=0.0008), mode="raw")
    _, _, share_trap = bt.attribution(trap)
    assert share_trap > 70.0, share_trap
    # 真 seasonality: 更分散,头名不应主导
    true = bt.backtest(make_panel(42, season=0.001), mode="raw")
    _, _, share_true = bt.attribution(true)
    assert share_true < 70.0, share_true

def test_bars_contract_path_is_offline_and_filters_half_days():
    rng = np.random.default_rng(123)
    tickers = [f"T{i}" for i in range(N_TICKERS)]
    dts = pd.bdate_range("2025-01-02", periods=60, tz="America/New_York")
    half_day = dts[10].date()
    bars = {}
    for ti, t in enumerate(tickers):
        idx = []
        ret = []
        for d in dts:
            nslots = 6 if d.date() == half_day else 13
            for k in range(nslots):
                idx.append(d + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=30*k))
                ret.append(rng.normal(0, 0.004) + (0.001 if ti == k % N_TICKERS else 0.0))
        ret = np.array(ret)
        bars[t] = pd.DataFrame({"Open": 100.0, "Close": 100.0*(1+ret)}, index=pd.DatetimeIndex(idx))
    lr = bt.keep_full_sessions(bt.to_slot_returns(bars))
    assert half_day not in set(lr["date"])
    assert lr.groupby(["ticker", "date"])["slot"].nunique().eq(13).all()
    m = bt.evaluate(lr, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS)
    assert m["n_test_days"] > 0, m

if __name__ == "__main__":
    for fn in (test_pure_noise_has_no_edge,
               test_true_seasonality_detected,
               test_trend_trap_is_flagged_as_momentum,
               test_genuine_seasonality_is_regime_robust,
               test_regime_momentum_trap_collapses,
               test_attribution_flags_single_name_dominance,
               test_bars_contract_path_is_offline_and_filters_half_days):
        fn(); print("PASS", fn.__name__)
    print("ALL GUARDRAILS PASS")
