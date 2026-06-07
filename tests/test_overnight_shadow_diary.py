import numpy as np
import pandas as pd

import overnight_shadow_diary as diary


def _bars():
    idx = pd.bdate_range("2026-01-01", periods=50)
    return {
        "A": pd.DataFrame({"Open": np.linspace(100, 105, len(idx)),
                           "Close": np.linspace(101, 106, len(idx))}, index=idx),
        "B": pd.DataFrame({"Open": np.linspace(50, 48, len(idx)),
                           "Close": np.linspace(49, 47, len(idx))}, index=idx),
        "C": pd.DataFrame({"Open": np.linspace(30, 30, len(idx)),
                           "Close": np.linspace(30, 30, len(idx))}, index=idx),
    }


def test_plan_settle_report_lifecycle(tmp_path):
    bars = _bars()
    plan = diary.create_plan(
        bars, rule="mean", lookback=10, top_n=2, universe=["A", "B", "C"],
        notional=1000.0, plan_date="2026-03-12"
    )
    assert len(plan) == 2
    assert abs(plan["notional"].sum() - 1000.0) < 1e-9
    assert plan["plan_id"].nunique() == 1

    all_plans = diary._append_csv(plan, tmp_path / diary.PLAN_FILE, ["plan_id", "ticker"])
    assert len(all_plans) == 2
    all_plans = diary._append_csv(plan, tmp_path / diary.PLAN_FILE, ["plan_id", "ticker"])
    assert len(all_plans) == 2

    detail, summary = diary.settle_plan(plan, bars, cost_bps=1.0)
    assert len(detail) == 2
    assert summary["status"].iloc[0] in {"settled", "pending"}
    assert "net_bps" in summary.columns

    settlements = diary._append_csv(summary, tmp_path / diary.SETTLE_FILE, ["plan_id"])
    report = diary.rolling_report(settlements)
    if summary["status"].iloc[0] == "settled":
        assert not report.empty


def test_pending_settlement_when_next_open_missing():
    bars = _bars()
    plan = diary.create_plan(bars, rule="mean", lookback=10, top_n=1, universe=["A"], plan_date="2026-03-12")
    # plan history_end is the last row, so there is no next open yet.
    detail, summary = diary.settle_plan(plan, bars, cost_bps=1.0)
    assert summary["status"].iloc[0] == "pending"
    assert detail["status"].iloc[0] == "pending_next_open"
