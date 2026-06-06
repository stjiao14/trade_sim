import numpy as np
import pandas as pd

import factor_xray as fx


def _panel(seed, mode):
    rng = np.random.default_rng(seed)
    n = 800
    if mode == "fake":          # All assets load on one common factor.
        F = rng.normal(0, 0.015, n)
        R = pd.DataFrame({"A": 1.0 * F + rng.normal(0, 0.005, n),
                          "B": 0.9 * F + rng.normal(0, 0.004, n),
                          "C": 1.1 * F + rng.normal(0, 0.005, n),
                          "D": 3.0 * F + rng.normal(0, 0.015, n)})
        w = [0.1, 0.45, 0.3, 0.15]
    else:                       # Mutually independent assets.
        R = pd.DataFrame({f"A{i}": rng.normal(0, 0.01, n) for i in range(5)})
        w = [0.2] * 5
    return R, w


def test_fake_diversification_flagged():
    d = fx.diversification(*_panel(1, "fake"))
    assert d["ENB"] < 1.6 and d["PC1_share"] > 0.8 and d["DR"] < 1.2


def test_real_diversification_scores_high():
    d = fx.diversification(*_panel(1, "real"))
    assert d["ENB"] > 4.0 and d["DR"] > 1.8 and d["PC1_share"] < 0.35


def test_load_full_portfolio_maps_private_index_trusts(tmp_path):
    p = tmp_path / "holdings.csv"
    pd.DataFrame([
        dict(ticker_symbol="CUR:USD", name="US Dollar", subtype="cash", institution_value=10_000),
        dict(ticker_symbol="", name="Instl 500 Index Trust", subtype="", institution_value=100_000),
        dict(ticker_symbol="", name="Instl Ext Market Idx Tr", subtype="", institution_value=50_000),
        dict(ticker_symbol="GOOG", name="Alphabet RSU", subtype="rsu", institution_value=25_000),
    ]).to_csv(p, index=False)
    weights, notes = fx.load_full_portfolio(p)
    assert weights["CASH"] == 10_000
    assert weights["SPY"] == 100_000
    assert weights["VXF"] == 50_000
    assert weights["GOOGL"] == 25_000
    assert len(notes) == 3
