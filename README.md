# trade_sim

## How to falsify a new signal

Write a `signal_fn(hist) -> Series`, where `hist` is the lookback window for one slot/bucket
with rows as prior trading days and columns as units/tickers. Higher scores are selected first.

```python
from signal_lab import falsify, print_verdict


def my_signal(hist):
    return hist.mean()


v = falsify(panel, my_signal)
print_verdict(v)
```

A claim survives only if every gate passes: positive net after cost, beats the random floor,
not single-name dominated, not only high-vol-regime driven, survives dropping top-vol days,
and has a daily bootstrap CI above zero.

The synthetic controls in `tests/test_signal_lab.py` prove the gauntlet still distinguishes
真信号 / 噪声 / 单票动量 / regime 陷阱.
