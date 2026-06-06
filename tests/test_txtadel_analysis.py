import numpy as np
import pandas as pd

import txtadel_analysis as tx


SAMPLE = """
Txtadel
Date: 2026-06-04 15:40:12 Red Alert: LGTM
TickerWeight (%) Buy@CloseSell@OpenGain/Loss (%)
XLU 34.98% $43.94 $44.04 +0.23%
SMH 23.71% $627.53 $605.00 -3.59%
EMXC 14.30% $101.68 $97.60 -4.01%
XLE 14.17% $58.75 $58.70 -0.09%
GLD 12.84% $411.27 $404.26 -1.70%
Total Return:-1.58%
Date: 2026-06-05 15:40:13 Red Alert: LGTM
TickerWeight (%) Buy@CloseSell@OpenGain/Loss (%)
XLU 35.00% $44.36 Pending Pending
CGDV 22.11% $47.86 Pending Pending
Total Return: Pending
"""


def test_parse_txtadel_text_and_recompute_return():
    orders, daily = tx.parse_txtadel_text(SAMPLE)
    assert len(orders) == 7
    assert daily.loc[daily["date"].astype(str).eq("2026-06-04"), "posted_total_pct"].iloc[0] == -1.58
    xlu = orders[(orders["ticker"] == "XLU") & (orders["date"].astype(str) == "2026-06-04")].iloc[0]
    assert xlu["weight_pct"] == 34.98
    assert xlu["buy_close"] == 43.94
    assert xlu["sell_open"] == 44.04
    cmp = tx.compare_posted_vs_recomputed(orders, daily)
    row = cmp[cmp["date"].astype(str) == "2026-06-04"].iloc[0]
    manual = 34.98 * 0.23 / 100 + 23.71 * -3.59 / 100 + 14.30 * -4.01 / 100 + 14.17 * -0.09 / 100 + 12.84 * -1.70 / 100
    assert abs(row["calc_total_pct"] - manual) < 1e-12
    assert abs(row["diff_pct"] - (manual - (-1.58))) < 1e-12


def test_capped_normalize_respects_cap_and_sum():
    w = tx.capped_normalize(pd.Series({"A": 100.0, "B": 10.0, "C": 10.0, "D": 10.0}), cap=0.35)
    assert abs(w.sum() - 1.0) < 1e-12
    assert w.max() <= 0.3500000001
    assert w.idxmax() == "A"


def test_inverse_vol_fit_identifies_matching_lookback():
    dates = pd.bdate_range("2026-01-01", periods=90)
    rng = np.random.default_rng(0)
    sig = pd.Series({"XLU": 0.006, "SMH": 0.012, "EMXC": 0.016, "XLE": 0.018, "GLD": 0.020})
    returns = pd.DataFrame({c: rng.normal(0, sig[c], len(dates)) for c in sig.index}, index=dates)
    asof = pd.Timestamp("2026-05-01")
    weights = tx.inverse_vol_weights(returns, sig.index, asof, lookback=60, cap=0.35)
    orders = pd.DataFrame([
        dict(date=asof.date(), ticker=t, weight_pct=float(w * 100), buy_close=100.0, sell_open=101.0, gain_pct=1.0)
        for t, w in weights.items()
    ])
    fit = tx.fit_inverse_vol_weighting(orders, returns, lookbacks=(20, 60), cap=0.35)
    best = fit.sort_values("mae_pct").iloc[0]
    assert int(best["lookback"]) == 60
    assert best["mae_pct"] < 1e-9
