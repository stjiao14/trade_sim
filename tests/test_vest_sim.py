import numpy as np

import vest_diversify_sim as V


CFG = dict(goog_price=250., rsu_unvested_usd=566_470., liquid_goog_usd=45_996.,
           liquid_basket_usd=893_370., vest_months=list(range(3, 49, 3)))


def test_diversify_reduces_risk_at_equal_mu():
    r = V.simulate(CFG, 0.07, 0.07, sg=0.32, sb=0.16, rho=0.60, seed=0)
    assert r["sell"].std() < r["hold"].std()                    # 分散降方差
    assert r["sell_dd"].mean() < r["hold_dd"].mean()            # 分散降回撤
    assert np.median(r["sell"]) >= np.median(r["hold"]) - 1e-6  # 同 mu 下 SELL 中位不输


def test_breakeven_drift_is_positive_and_sane():
    e = V.breakeven_drift(CFG, 0.07, 0.32, 0.16, 0.60)
    assert 0.0 < e < 0.10                                       # 需要正超额,且量级合理
