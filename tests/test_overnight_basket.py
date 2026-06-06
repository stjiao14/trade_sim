import numpy as np
import pandas as pd

import overnight_basket_backtest as ob


def _bars():
    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    return {
        "AAA": pd.DataFrame({"Open": [100.0, 102.0, 103.0, 105.0],
                             "Close": [101.0, 103.0, 104.0, 106.0]}, index=idx),
        "BBB": pd.DataFrame({"Open": [50.0, 49.0, 50.0, 51.0],
                             "Close": [49.5, 50.0, 51.0, 52.0]}, index=idx),
    }


def test_overnight_returns_use_next_open_over_prior_close():
    r = ob.overnight_returns(_bars())
    assert list(r.columns) == ["AAA", "BBB"]
    assert list(r.index) == list(pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]))
    assert abs(r.loc[pd.Timestamp("2026-01-02"), "AAA"] - (102.0 / 101.0 - 1.0)) < 1e-12
    assert abs(r.loc[pd.Timestamp("2026-01-05"), "BBB"] - (50.0 / 50.0 - 1.0)) < 1e-12


def test_static_basket_backtest_applies_cost_and_weights():
    r = ob.overnight_returns(_bars())
    res = ob.backtest_static_basket(r, weights={"AAA": 0.75, "BBB": 0.25}, cost_bps=2.0)
    first_gross = 0.75 * (102.0 / 101.0 - 1.0) + 0.25 * (49.0 / 49.5 - 1.0)
    assert len(res) == 3
    assert abs(res.loc[0, "gross"] - first_gross) < 1e-12
    assert abs(res.loc[0, "net"] - (first_gross - 2.0 / 1e4)) < 1e-12
    assert res["n_assets"].min() == 2


def test_static_basket_renormalizes_when_asset_missing():
    r = ob.overnight_returns(_bars())
    r.loc[pd.Timestamp("2026-01-05"), "BBB"] = np.nan
    res = ob.backtest_static_basket(r, weights={"AAA": 0.5, "BBB": 0.5}, cost_bps=0.0)
    row = res[res["date"].eq(pd.Timestamp("2026-01-05"))].iloc[0]
    assert row["n_assets"] == 1
    assert abs(row["gross"] - r.loc[pd.Timestamp("2026-01-05"), "AAA"]) < 1e-12


def test_performance_metrics_are_finite_for_nonempty_result():
    r = ob.overnight_returns(_bars())
    res = ob.backtest_static_basket(r)
    m = ob.performance_metrics(res)
    assert m["n_days"] == 3
    assert np.isfinite(m["total_return_pct"])
    assert np.isfinite(m["mean_bps"])
