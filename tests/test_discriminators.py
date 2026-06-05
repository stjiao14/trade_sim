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

if __name__ == "__main__":
    for fn in (test_pure_noise_has_no_edge,
               test_true_seasonality_detected,
               test_trend_trap_is_flagged_as_momentum):
        fn(); print("PASS", fn.__name__)
    print("ALL GUARDRAILS PASS")
