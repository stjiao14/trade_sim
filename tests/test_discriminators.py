"""Synthetic regression guardrails for no-edge / true-seasonality / momentum-trap verdicts.
Any change that breaks these tests likely damaged the discriminator logic."""
import numpy as np, pandas as pd
import intraday_seasonality_backtest as bt

N_TICKERS, LOOKBACK, COST_RT_BPS = 9, 30, 3.0

def make_panel(seed, season=0.0, trend=0.0, n=N_TICKERS, days=60):
    """Build a synthetic half-hour return panel.

    season=true stock x slot edge; trend=persistent per-ticker drift.
    Fixed RNG draw order keeps results deterministic.
    """
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
    return bt.evaluate(make_panel(42, **kw), lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS,
                       random_seeds=8)

def test_pure_noise_has_no_edge():
    m = _eval(season=0.0, trend=0.0)
    assert m["season_excess_bps"] < 3.0,  m   # Not true seasonality; near selection-bias floor.
    assert m["raw_net_bps"]      < 0.0,  m    # Not tradeable after cost.
    assert m["concentration_pct"] < 30.0, m   # No momentum concentration.

def test_true_seasonality_detected():
    m = _eval(season=0.001, trend=0.0)
    assert m["season_excess_bps"] > 4.0,  m   # Isolated signal clears the floor.
    assert m["concentration_pct"] < 30.0, m   # Picks are distributed across tickers/slots.

def test_trend_trap_is_flagged_as_momentum():
    m = _eval(season=0.0, trend=0.0008)
    assert m["raw_net_bps"]      > 0.0,  m    # Looks profitable, which is the trap.
    assert m["season_excess_bps"] < 3.0, m    # Seasonality is still an illusion.
    assert m["concentration_pct"] > 35.0, m   # Momentum contamination is flagged.

def make_true_uniform(seed=42, season=0.001, n=9, days=60):
    """Uniform true seasonality, not regime-driven."""
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
    """Pure regime momentum trap: relative drift pays only on a few high-vol days."""
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
    reg = bt.season_excess_by_regime(lr, seeds=8)
    # Drop top-5 high-vol days; true signal should remain positive/significant.
    mean5, t5, _ = dvd[5]
    assert mean5 > 0 and t5 > 1.0, dvd
    # Isolated seasonality should be positive in both high/low regimes.
    assert reg["high"] > 2.0 and reg["low"] > 2.0, reg

def test_regime_momentum_trap_collapses():
    lr = make_regime_trap()
    res = bt.backtest(lr, mode="raw")
    dvd = bt.drop_top_vol_days(res, lr)
    reg = bt.season_excess_by_regime(lr, seeds=8)
    # After dropping high-vol days, the edge should no longer be significant.
    _, t5, _ = dvd[5]
    assert t5 < 1.0, dvd
    # Isolated seasonality should not be true in both regimes.
    assert not (reg["high"] > 2.0 and reg["low"] > 2.0), reg

def test_attribution_flags_single_name_dominance():
    # Trend trap: one ticker should dominate P&L.
    trap = bt.backtest(make_panel(42, trend=0.0008), mode="raw")
    _, _, share_trap = bt.attribution(trap)
    assert share_trap > 70.0, share_trap
    # True seasonality should be more diversified; top ticker should not dominate.
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

def test_long_short_runs_and_amplifies_genuine_seasonality():
    lr = make_panel(42, season=0.001)
    lo = bt.backtest(lr, mode="raw")
    ls = bt.backtest_ls(lr)
    assert len(ls) == len(lo), (len(ls), len(lo))          # One trade per day/slot.
    assert ls["net"].mean() > 0, ls["net"].mean()          # Neutral version is positive under true signal.
    assert ls["net"].mean() * 1e4 > lo["net"].mean() * 1e4 # Both tails contribute, so LS beats long-only.

if __name__ == "__main__":
    for fn in (test_pure_noise_has_no_edge,
               test_true_seasonality_detected,
               test_trend_trap_is_flagged_as_momentum,
               test_genuine_seasonality_is_regime_robust,
               test_regime_momentum_trap_collapses,
               test_attribution_flags_single_name_dominance,
               test_bars_contract_path_is_offline_and_filters_half_days,
               test_long_short_runs_and_amplifies_genuine_seasonality):
        fn(); print("PASS", fn.__name__)
    print("ALL GUARDRAILS PASS")
