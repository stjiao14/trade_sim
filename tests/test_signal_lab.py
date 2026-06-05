import numpy as np

import intraday_seasonality_backtest as bt
import signal_lab as L
from tests.test_discriminators import make_panel, make_regime_trap


def test_reproduces_backtest_raw():
    lr = make_panel(42, season=0.001)
    a = L.run(lr, L.seasonality_signal)
    b = bt.backtest(lr, mode="raw")
    assert len(a) == len(b)
    assert (a["pick"].values == b["pick"].values).all()
    assert np.abs(a["net"].values - b["net"].values).max() < 1e-12


def test_verdict_true_signal_passes():
    assert L.falsify(make_panel(42, season=0.001), L.seasonality_signal)["verdict"] == "PASS"


def test_verdict_noise_fails():
    assert L.falsify(make_panel(1, season=0.0), L.seasonality_signal)["verdict"] == "FAIL"


def test_verdict_trend_trap_fails_on_concentration():
    v = L.falsify(make_panel(42, trend=0.0008), L.seasonality_signal)
    assert v["verdict"] == "FAIL" and "not_single_name" in v["fail_reasons"]


def test_verdict_regime_trap_fails_on_fragility():
    v = L.falsify(make_regime_trap(7), L.seasonality_signal)
    assert v["verdict"] == "FAIL"
    assert ("survives_drop_top_days" in v["fail_reasons"]) or ("regime_robust" in v["fail_reasons"])


def test_other_signals_run_through_battery():
    lr = make_panel(42, season=0.001)
    for fn in (L.momentum_signal, L.reversal_signal):
        v = L.falsify(lr, fn)
        assert v["verdict"] in ("PASS", "FAIL") and set(v["gates"])
