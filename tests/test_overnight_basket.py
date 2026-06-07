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


def test_regime_panel_computes_macro_roc():
    idx = pd.bdate_range("2026-01-01", periods=5)
    bars = {
        "^VIX": pd.DataFrame({"Open": [20, 20, 20, 20, 20], "Close": [18, 22, 31, 25, 19]}, index=idx),
        "XLY": pd.DataFrame({"Open": [1, 1, 1, 1, 1], "Close": [100, 110, 120, 90, 130]}, index=idx),
        "XLP": pd.DataFrame({"Open": [1, 1, 1, 1, 1], "Close": [100, 100, 100, 100, 100]}, index=idx),
    }
    reg = ob.regime_panel(bars, macro_lookback=2)
    expected = (120 / 100 - 100 / 100) / (120 / 100)
    assert abs(reg.loc[idx[2], "macro_roc"] - expected) < 1e-12


def test_apply_regime_gate_uses_lagged_decision_data():
    r = ob.overnight_returns(_bars())
    res = ob.backtest_static_basket(r, cost_bps=0.0)
    regime = pd.DataFrame(
        {"vix": [20.0, 35.0, 20.0], "macro_roc": [0.0, 0.0, -0.5]},
        index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
    )
    same_day = ob.apply_regime_gate(res, regime, vix_max=30, macro_min=-0.2, decision_lag=0)
    lagged = ob.apply_regime_gate(res, regime, vix_max=30, macro_min=-0.2, decision_lag=1)
    assert list(same_day["date"]) == [pd.Timestamp("2026-01-02")]
    assert list(lagged["date"]) == [pd.Timestamp("2026-01-05")]


def test_selection_rule_backtest_selects_history_only_winner():
    idx = pd.bdate_range("2026-01-01", periods=8)
    signal = pd.DataFrame({
        "A": [0.01, 0.01, 0.01, 0.01, -0.50, -0.50, -0.50, -0.50],
        "B": [0.00, 0.00, 0.00, 0.00, 0.90, 0.90, 0.90, 0.90],
    }, index=idx)
    overnight = pd.DataFrame({"A": [0.02] * 8, "B": [-0.01] * 8}, index=idx)
    res = ob.backtest_selection_rule(overnight, signal, rule="mean", lookback=4, top_n=1, cost_bps=0.0)
    first = res.iloc[0]
    assert first["date"] == idx[4]
    assert first["picks"] == "A"
    assert abs(first["net"] - 0.02) < 1e-12


def test_grid_oos_shadow_and_signal_lab_bridge():
    r = ob.overnight_returns(_bars())
    grid = ob.summarize_rule_grid(r, r, rules=("mean", "reversal"), lookbacks=(2,), top_ns=(1,), cost_bps=0.0)
    assert set(grid["rule"]) == {"mean", "reversal"}
    oos = ob.walk_forward_oos_grid(
        r, r, [dict(rule="mean", lookback=2, top_n=1, weighting="equal", cost_bps=0.0)], freq="Y"
    )
    assert not oos.empty
    plan = ob.next_selection_plan(_bars(), rule="mean", lookback=2, top_n=1, notional=1000.0)
    assert not plan.empty and abs(plan["notional"].sum() - 1000.0) < 1e-12
    panel = ob.to_signal_lab_panel(r)
    assert set(panel.columns) == {"date", "slot", "ticker", "ret"}
    fn = ob.overnight_signal_fn("mean")
    wide = panel.pivot_table(index="date", columns="ticker", values="ret")
    assert not fn(wide).empty


def test_cost_model_uses_weighted_per_ticker_cost():
    w = pd.Series({"A": 0.75, "B": 0.25})
    c = ob.realized_cost_bps(w, cost_bps=1.0, cost_by_ticker={"A": 2.0, "B": 6.0})
    assert abs(c - 3.0) < 1e-12


def test_basket_falsifier_structure_and_random_floor():
    idx = pd.bdate_range("2026-01-01", periods=30)
    signal = pd.DataFrame({
        "A": np.linspace(0.01, 0.02, len(idx)),
        "B": np.zeros(len(idx)),
        "C": -np.linspace(0.01, 0.02, len(idx)),
    }, index=idx)
    overnight = pd.DataFrame({
        "A": np.full(len(idx), 0.01),
        "B": np.zeros(len(idx)),
        "C": np.full(len(idx), -0.01),
    }, index=idx)
    v = ob.falsify_selection_rule(
        overnight, signal, rule="mean", lookback=5, top_n=1,
        cost_bps=0.0, random_seeds=3, bootstrap_n=200, drop_ks=(0, 1)
    )
    assert v["verdict"] in ("PASS", "FAIL")
    assert v["net_bps"] > v["random_floor_bps"]
    assert set(v["gates"])


def test_drop_top_move_days_and_attribution():
    r = ob.overnight_returns(_bars())
    res = ob.backtest_selection_rule(r, r, rule="mean", lookback=2, top_n=1, cost_bps=0.0)
    attr, pick_share, pnl_share = ob.basket_attribution(res)
    drops = ob.drop_top_move_days(res, r, ks=(0, 1))
    assert not attr.empty
    assert pick_share >= 0
    assert np.isfinite(pnl_share)
    assert 0 in drops and 1 in drops
