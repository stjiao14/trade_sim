# trade_sim

这个仓库现在有两台机器。

第一台是 **信号证伪机**:把任何交易信号放进同一套反证关卡,看它到底是可重复 edge,还是成本、噪声、动量、单季度 regime 的幻觉。

第二台是 **个人风险 X 光机**:从本地持仓 CSV 出发,看真实 GOOGL 穿透、全组合因子暴露、独立押注数,以及 RSU 归属后持有 vs 卖出分散的风险分布。

真实金额和 API key 不进仓库。真实持仓放本地 CSV 或 git-ignored 的 `config_local.py`;仓库只提交 `config_example.py` 的 dummy 数字。

## Setup

```bash
pip install numpy pandas yfinance pytest
```

可选:长历史 intraday 回测用 Polygon。

```bash
export POLYGON_API_KEY="..."
```

Windows PowerShell:

```powershell
$env:POLYGON_API_KEY="..."
```

推荐在 `config_local.py` 里统一放本地持仓 CSV 路径:

```python
HOLDINGS_CSV = r"C:\path\to\your_holdings.csv"
```

`config_local.py` 已被 `.gitignore` 忽略。

## Quick Test

```bash
pytest -q
```

测试包含:

- intraday seasonality 的三类合成 guardrail:无 edge / 真 seasonality / 动量陷阱
- regime fragility、归因、long/short、Polygon 数据 contract
- `signal_lab` 通用信号证伪
- paper trading 本地 broker / 风控 / Alpaca paper adapter 的离线测试
- GOOGL 穿透、factor x-ray、RSU 分散模拟的离线数学测试

## A. 信号证伪机

### `intraday_seasonality_backtest.py`

这是原始 intraday seasonality 策略的真实数据回测。策略每天对 13 个 30 分钟槽分别选一只股票,核心疑问不是“能不能回测赚钱”,而是:

- 是否真有 cross-sectional seasonality
- 扣掉 realistic round-trip cost 后是否可交易
- 是否只是动量、单票集中、少数高波动日伪装出来的 edge

运行 yfinance 60 天版本:

```bash
python intraday_seasonality_backtest.py
```

运行 Polygon 18 个月多 regime 版本:

```python
import intraday_seasonality_backtest as bt
bt.long_window_report_polygon()
```

核心发现:这个 intraday seasonality 策略已经被长窗口结果证伪。18 个月全样本 raw net 约为负,seasonality excess 接近随机地板;只有 2026Q2 一个短窗口出现正收益,且集中度高、regime 特征明显。它更像单季度行情/动量污染,不是稳定可交易的 seasonality edge。

### `signal_lab.py`

通用信号证伪库。写一个 `signal_fn(hist) -> Series`,就能跑同一套关卡:

```python
from signal_lab import falsify, print_verdict

def my_signal(hist):
    return hist.mean()

v = falsify(panel, my_signal)
print_verdict(v)
```

`hist` 是某个 slot/bucket 的 lookback 面板:行是过去交易日,列是 ticker/unit,值是收益。分数越高越优先选。

内置信号:

- `seasonality_signal`:同槽均值,复现原版 backtest
- `momentum_signal`:lookback 累计收益
- `reversal_signal`:lookback 累计收益反转

一个信号只有全部 gate 都通过才算 PASS:扣成本后为正、超过随机地板、非单票主导、不是只在高波动日赚钱、抽掉极端日仍成立、按天 bootstrap CI 下界大于 0。

### `paper_broker.py` / `paper_runner.py`

纸面 forward-test 执行层。它不证明 alpha,只把已经通过研究关卡的订单意图变成可审计的 paper orders / fills / positions。

离线本地纸面账户:

```python
from paper_broker import LocalPaperBroker, OrderIntent, RiskGate, RiskLimits

broker = LocalPaperBroker(cash=100_000, price_map={"AAPL": 200.0}, slippage_bps=1.0)
gate = RiskGate(RiskLimits(max_order_notional=5_000, max_symbol_notional=20_000))
broker.submit_order(OrderIntent("AAPL", "buy", notional=1_000, reason="demo"), risk_gate=gate)
print(broker.positions_frame())
```

把回测/信号结果转成订单:

```python
from paper_runner import intents_from_picks, run_intents, print_broker_report

intents = intents_from_picks(backtest_result, notional_per_trade=1_000)
run_intents(broker, intents, risk_gate=gate)
print_broker_report(broker)
```

真实 paper API 目前提供 `AlpacaPaperBroker`,只允许连接 `https://paper-api.alpaca.markets`,避免误连 live。需要环境变量 `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`。Polygon/Massive 适合行情数据;订单模拟和 paper execution 需要 broker API,所以两者角色不同。

## B. 个人风险 X 光机

### `concentration_analysis.py`

计算 Alphabet 真穿透敞口:直接 GOOG/GOOGL、SPY/QQQ/TQQQ look-through、未归属 RSU,以及可选人力资本。

```bash
python concentration_analysis.py path/to/holdings.csv
```

或在 `config_local.py` 设置 `HOLDINGS_CSV`,然后:

```bash
python concentration_analysis.py
```

输出告诉你:

- 流动账户 GOOGL 等价敞口占比
- 含 RSU 后的 GOOGL 等价敞口占比
- GOOGL -30% / -50% 情景损失
- GOOGL/SPY/QQQ/TQQQ 这条科技/指数 sleeve 对 GOOGL 的 beta/R²

注意:这里的 beta/R² 不是全账户因子回归;全账户请看 `factor_xray.py`。

### `factor_xray.py`

全组合因子 X 光 + 真分散度评分。

```bash
python factor_xray.py path/to/holdings.csv
```

或:

```bash
python factor_xray.py
```

它会输出:

- 因子代理回归:市场、科技、价值、规模、国际、利率、黄金、油
- R² 与共线性提示
- ENB:实际约几个独立押注
- DR:分散比,1.0 表示几乎没有分散收益
- PC1 占比:最大共同因子解释多少横截面方差
- 风险贡献表:每个标的贡献多少组合方差

读法:名义持有很多 ticker 不等于真分散。ENB、DR、PC1 和风险贡献表会把“看起来分散”翻译成“实际几个押注”。

### `vest_diversify_sim.py`

RSU 归属后 HOLD vs SELL 分散的 Monte Carlo。

```bash
python vest_diversify_sim.py
```

配置放在 `config_local.py`:

```python
VEST_HOLDINGS = dict(
    goog_price=250.0,
    rsu_unvested_usd=0.0,
    liquid_goog_usd=0.0,
    liquid_basket_usd=0.0,
)
VEST_MONTHS = list(range(3, 49, 3))
BASKET_PROXY = "VT"
N_PATHS = 20000
```

脚本默认假设 `mu_GOOG == mu_basket`,用来隔离风险差异,不是预测谁涨得更多。输出 HOLD vs SELL 的终值分布、标准差、最大回撤,以及 breakeven drift:GOOG 每年需要跑赢篮子多少,继续 HOLD 的中位数才追平 SELL。

## Privacy

- 不提交真实持仓 CSV
- 不提交 `config_local.py`
- 不提交 API key
- `config_example.py` 只放 dummy 数字

## Caveats

所有脚本都是研究和风险量化工具,不构成投资建议。因子代理 ETF 之间有共线性,重点看 R²、主导暴露和稳定性,不要把每个 beta 当成正交真因子。RSU 模拟用 GBM 简化世界,没有肥尾和 regime;如果要更严谨,下一步可以换 block-bootstrap 或 Student-t。
