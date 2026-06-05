import concentration_analysis as ca
import pandas as pd


def test_lookthrough_math():
    H = dict(googl_shares=500, rsu_unvested_shares=800, spy_usd=120_000, qqq_usd=80_000, tqqq_usd=40_000)
    px = dict(GOOGL=190.0, SPY=600.0, QQQ=530.0, TQQQ=90.0)
    rows, lg, nw, rsu = ca.compute_lookthrough(H, px, w_spy=0.038, w_qqq=0.050)
    assert abs(lg   - (95_000 + 4_560 + 4_000 + 6_000)) < 1e-6, lg     # 109,560
    assert abs(nw   - (95_000 + 120_000 + 80_000 + 40_000)) < 1e-6, nw  # 335,000
    assert abs(rsu  - 800*190.0) < 1e-6                                  # 152,000
    assert abs(lg/nw - 0.3270) < 1e-3, lg/nw                            # ~32.7%


def test_csv_mapping_offline(tmp_path):
    p = tmp_path / "holdings.csv"
    pd.DataFrame([
        dict(ticker_symbol="GOOG", name="Alphabet RSU", subtype="rsu", institution_value=190_000),
        dict(ticker_symbol="", name="Instl 500 Index Trust", subtype="", institution_value=100_000),
        dict(ticker_symbol="QQQM", name="Invesco NASDAQ 100 ETF", subtype="etf", institution_value=80_000),
        dict(ticker_symbol="TQQQ", name="ProShares UltraPro QQQ", subtype="etf", institution_value=20_000),
        dict(ticker_symbol="AMD", name="Advanced Micro Devices", subtype="common stock", institution_value=10_000),
    ]).to_csv(p, index=False)
    H, notes, _ = ca.load_holdings_from_csv(p, dict(GOOGL=190.0))
    assert abs(H["rsu_unvested_shares"] - 1000) < 1e-6
    assert H["spy_usd"] == 100_000
    assert H["qqq_usd"] == 80_000
    assert H["tqqq_usd"] == 20_000
    assert H["total_liquid_usd"] == 210_000
