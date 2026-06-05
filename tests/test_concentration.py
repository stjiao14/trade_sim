import concentration_analysis as ca


def test_lookthrough_math():
    H = dict(googl_shares=500, rsu_unvested_shares=800, spy_usd=120_000, qqq_usd=80_000, tqqq_usd=40_000)
    px = dict(GOOGL=190.0, SPY=600.0, QQQ=530.0, TQQQ=90.0)
    rows, lg, nw, rsu = ca.compute_lookthrough(H, px, w_spy=0.038, w_qqq=0.050)
    assert abs(lg   - (95_000 + 4_560 + 4_000 + 6_000)) < 1e-6, lg     # 109,560
    assert abs(nw   - (95_000 + 120_000 + 80_000 + 40_000)) < 1e-6, nw  # 335,000
    assert abs(rsu  - 800*190.0) < 1e-6                                  # 152,000
    assert abs(lg/nw - 0.3270) < 1e-3, lg/nw                            # ~32.7%
