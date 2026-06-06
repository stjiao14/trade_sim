"""Generic signal falsification lab: plug any signal_fn into the same gate battery."""
import numpy as np
import pandas as pd

from intraday_seasonality_backtest import (
    COST_RT_BPS,
    LOOKBACK,
    _market_move,
    _pivot,
    attribution,
    bootstrap_daily,
    daily_net_bps,
    drop_top_vol_days,
    regime_split,
)


def seasonality_signal(hist):
    """Same-slot mean, reproducing current backtest(mode='raw') scoring."""
    return hist.mean()


def momentum_signal(hist):
    """Lookback cumulative-return momentum."""
    return (1 + hist).prod() - 1


def reversal_signal(hist):
    """Lookback cumulative-return reversal."""
    return -((1 + hist).prod() - 1)


def run(lr, signal_fn, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS, select="argmax", seed=0):
    """Walk-forward picker using signal_fn. seasonality_signal matches backtest(raw) trade by trade."""
    piv = _pivot(lr); dates = sorted(lr["date"].unique()); slots = sorted(lr["slot"].unique())
    cols = list(piv.columns); dlev = piv.index.get_level_values("date")
    rng = np.random.default_rng(seed); rows = []
    for i in range(lookback, len(dates)):
        d = dates[i]; sub = piv[dlev.isin(set(dates[i-lookback:i]))]
        for s in slots:
            tr = sub.xs(s, level="slot")
            if tr.empty: continue
            sc = signal_fn(tr).dropna()
            if sc.empty: continue
            pick = sc.idxmax() if select == "argmax" else cols[rng.integers(len(cols))]
            try: rz = piv.loc[(d, s), pick]
            except KeyError: continue
            if pd.isna(rz): continue
            rows.append(dict(date=d, slot=s, pick=pick, gross=rz, net=rz - cost_rt_bps/1e4))
    return pd.DataFrame(rows)


def run_ls(lr, signal_fn, lookback=LOOKBACK, cost_rt_bps=COST_RT_BPS):
    """Neutral variant: long argmax and short argmin; net=(r_long-r_short)-2*cost."""
    piv = _pivot(lr); dates = sorted(lr["date"].unique()); slots = sorted(lr["slot"].unique())
    dlev = piv.index.get_level_values("date"); rows = []
    for i in range(lookback, len(dates)):
        d = dates[i]; sub = piv[dlev.isin(set(dates[i-lookback:i]))]
        for s in slots:
            tr = sub.xs(s, level="slot"); sc = signal_fn(tr).dropna()
            if len(sc) < 2: continue
            L, S = sc.idxmax(), sc.idxmin()
            try: rl, rs = piv.loc[(d, s), L], piv.loc[(d, s), S]
            except KeyError: continue
            if pd.isna(rl) or pd.isna(rs): continue
            rows.append(dict(date=d, slot=s, long=L, short=S, net=(rl-rs)-2*cost_rt_bps/1e4))
    return pd.DataFrame(rows)


def excess_over_random(lr, signal_fn, lookback=LOOKBACK, seeds=20):
    """Generic excess over random floor: argmax net minus random net, in bps/trade."""
    a = run(lr, signal_fn, lookback=lookback, select="argmax")["net"].mean()
    f = np.mean([run(lr, signal_fn, lookback=lookback, select="random", seed=k)["net"].mean()
                 for k in range(seeds)])
    return float((a - f) * 1e4)


def falsify(lr, signal_fn, lookback=LOOKBACK, conc_max=30.0, drop_ks=(0,1,3,5),
            random_seeds=20, bootstrap_n=10000):
    """Run all gates and return metrics, gate booleans, PASS/FAIL verdict, and reasons."""
    res = run(lr, signal_fn, lookback=lookback)
    net = res["net"].mean() * 1e4
    conc = res["pick"].value_counts(normalize=True).iloc[0] * 100
    _, _, top_share = attribution(res)
    hi, lo = regime_split(lr, res)
    dvd = drop_top_vol_days(res, lr, ks=drop_ks)
    k_last = max(drop_ks); mean_k, t_k, _ = dvd[k_last]
    daily = daily_net_bps(res); dm, dlo, dhi = bootstrap_daily(daily, n=bootstrap_n)
    xs = excess_over_random(lr, signal_fn, lookback=lookback, seeds=random_seeds)
    ls_net = run_ls(lr, signal_fn, lookback=lookback)["net"].mean() * 1e4

    gates = {
        "net_after_cost_positive": net > 0,
        "beats_random_floor":      xs > 0,
        "not_single_name":         (conc < conc_max) and (top_share < 70.0),
        "regime_robust":           not (hi > 0 and lo <= 0),          # Cannot earn only on high-vol days.
        "survives_drop_top_days":  (mean_k > 0) and (t_k > 1.0),       # Still significant after dropping extreme days.
        "daily_ci_excludes_zero":  dlo > 0,                            # Daily bootstrap CI lower bound is above zero.
    }
    reasons = [k for k, ok in gates.items() if not ok]
    return dict(
        verdict="PASS" if not reasons else "FAIL",
        fail_reasons=reasons,
        net_bps=float(net), excess_over_random_bps=xs, concentration_pct=float(conc),
        top_ticker_share_pct=float(top_share), regime_hi_bps=float(hi), regime_lo_bps=float(lo),
        drop_top_days=dvd, daily_mean_bps=dm, daily_ci=(dlo, dhi), ls_net_bps=float(ls_net),
        gates=gates,
    )


def print_verdict(v):
    """Print a readable falsify() verdict and key metrics."""
    print(f"VERDICT: {v['verdict']}")
    if v["fail_reasons"]:
        print("Failed gates: " + ", ".join(v["fail_reasons"]))
    else:
        print("All gates passed.")
    print(f"Net/trade: {v['net_bps']:+.2f} bps | excess over random: {v['excess_over_random_bps']:+.2f} bps")
    print(f"Concentration: {v['concentration_pct']:.1f}% | top-name P&L share: {v['top_ticker_share_pct']:.1f}%")
    print(f"regime hi/lo: {v['regime_hi_bps']:+.2f}/{v['regime_lo_bps']:+.2f} bps")
    print(f"Daily mean: {v['daily_mean_bps']:+.2f} bps | bootstrap CI "
          f"[{v['daily_ci'][0]:+.2f}, {v['daily_ci'][1]:+.2f}]")
    print(f"long/short neutral: {v['ls_net_bps']:+.2f} bps/trade")


def falsify_long_window(lr, signal_fn, freq="Q", lookback=LOOKBACK):
    """Run falsify by calendar segment to check long-window regime persistence."""
    out = {}
    dts = pd.to_datetime(lr["date"])
    for p in sorted(dts.dt.to_period(freq).unique()):
        sub = lr[dts.dt.to_period(freq).eq(p)]
        try:
            out[str(p)] = falsify(sub, signal_fn, lookback=lookback)
        except Exception:
            continue
    return out
