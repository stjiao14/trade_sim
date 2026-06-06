# trade_sim

这个仓库现在有两台机器。

第一台是 **信号证伪机**:把任何交易信号放进同一套反证关卡,看它到底是可重复 edge,还是成本、噪声、动量、单季度 regime 的幻觉。

第二台是 **个人风险 X 光机**:从本地持仓 CSV 出发,看真实 GOOGL 穿透、全组合因子暴露、独立押注数,以及 RSU 归属后持有 vs 卖出分散的风险分布。

真实金额和 API key 不进仓库。真实持仓放本地 CSV 或 git-ignored 的 `config_local.py`;仓库只提交 `config_example.py` 的 dummy 数字。

## 功能地图

这个 repo 不是一个“自动赚钱 bot”,而是一套研究和风控工具。它的主线是:先证伪信号,再做纸面执行;先看真实集中度,再讨论是否分散。

| 模块 | 解决的问题 | 主要输出 | 需要 API |
|---|---|---|---|
| `intraday_seasonality_backtest.py` | 原始 intraday seasonality 策略是否真有 edge | 四道 gate、长窗口分季度诊断、long-only vs long/short | yfinance 可跑短窗;Polygon/Massive 跑长窗 |
| `signal_lab.py` | 任意新信号是否经得起同一套反证关卡 | `PASS/FAIL`、失败原因、随机地板、regime/集中度/CI | 不需要,只要你给 panel |
| `forward_paper.py` | 把研究信号变成下一交易日 paper 订单计划 | `paper_plan.csv`、`paper_orders.csv`、research gate 日志 | Polygon/Massive;可选 Alpaca paper |
| `paper_broker.py` | 本地撮合或 Alpaca paper 下单/回读 | fills、orders、positions、account snapshot | 可选 Alpaca paper |
| `broker_benchmark.py` | 比较 broker API 延迟和稳定性 | p50/p95/max latency、error rate | Alpaca paper |
| `concentration_analysis.py` | GOOGL/GOOG 真穿透敞口是多少 | 直接持股 + ETF look-through + RSU + 情景损失 | yfinance 持仓/价格;也可用 fallback 权重 |
| `factor_xray.py` | 全账户到底暴露在哪些系统性因子 | factor beta/R²、ENB、DR、PC1、风险贡献 | yfinance |
| `vest_diversify_sim.py` | RSU 归属后 HOLD vs SELL 的风险分布 | 分位数、标准差、最大回撤、breakeven drift | yfinance 校准;失败时有 fallback |
| `web_ui.py` | 不手写 `config_local.py`,用浏览器配置 key 和路径,并查看本地 monitor portal | 本地 git-ignored 配置文件、paper logs 状态、可选 Alpaca snapshot | 不需要;可选 Alpaca paper |

## 典型使用方式

### 1. 证伪一个交易信号

先把历史数据整理成 `DataFrame[date, slot, ticker, ret]`,再写一个 `signal_fn(hist) -> Series`:

```python
from signal_lab import falsify, print_verdict

def my_signal(hist):
    return hist.mean()

v = falsify(panel, my_signal)
print_verdict(v)
```

只有当扣成本后为正、超过随机地板、不过度集中、不是只靠高波动日、抽掉极端日仍成立、按天 bootstrap CI 下界大于 0 时,它才会 PASS。否则工具会告诉你具体死在哪个 gate。

### 2. 跑 intraday seasonality 长窗口复盘

短窗口可以用 yfinance 直接跑;真正判断 persistence 要用 Polygon/Massive 的 30m 长历史:

```python
import intraday_seasonality_backtest as bt
bt.long_window_report_polygon()
```

当前核心结论:这个 intraday seasonality 策略在 18 个月多 regime 长窗口里没有稳定通过。近期正收益集中在单季度/高波动 regime,并伴随较高单票集中度,更像动量或行情污染,不是可依赖的 standalone seasonality edge。

### 3. 纸面 forward test

研究 gate 先过,再进入 paper。默认 shadow 只写订单计划,不碰 broker:

```bash
python forward_paper.py --mode shadow --out paper_logs
```

如果要接 Alpaca paper,先用本地 Web UI 或环境变量配置 key,再显式指定 broker:

```bash
python forward_paper.py --broker alpaca-paper --mode paper --notional 1000 --require-pass --out paper_logs
```

`--require-pass` 是保险:研究闸门 FAIL 时只输出诊断,不执行 paper 订单。

### 4. 看个人账户风险

把真实持仓 CSV 路径放进 `config_local.py` 或 Web UI,然后分别跑:

```bash
python concentration_analysis.py
python factor_xray.py
python vest_diversify_sim.py
```

这三件事合起来回答:你对 GOOGL/科技到底有多少穿透敞口,名义上很多持仓实际约几个独立押注,以及 RSU 归属后继续 HOLD 需要 GOOG 每年多跑赢篮子多少才值得承担集中风险。

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

### 本地 Web UI 配置

如果不想手动编辑 `config_local.py`,可以启动一个只监听本机的配置页面:

```bash
python web_ui.py
```

Windows 上如果 `python` 被 Microsoft Store alias 抢走,可以直接用 Codex bundled Python:

```powershell
C:\Users\stjia\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe web_ui.py
```

然后打开:

```text
http://127.0.0.1:8765
```

这个页面可以配置 Alpaca paper key、Polygon key、持仓 CSV 路径、paper trading 风控参数。保存后会写入 git-ignored 的 `config_local.py`;页面只显示 `configured/missing` 状态,不会把已保存的 secret 回显出来。环境变量仍然优先于本地配置,所以临时覆盖 key 时可以继续用 `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `POLYGON_API_KEY`。

同一个本地服务还提供 monitor portal:

```text
http://127.0.0.1:8765/monitor
```

Monitor 默认只读本地 `paper_logs` CSV,不会触发下单。它会显示:

- API key 配置状态
- `paper_plan.csv` / `paper_orders.csv` / `paper_fills.csv` / `paper_summary.csv` / `paper_research.csv` 是否存在和新鲜度
- 最近 research gate 的 PASS/FAIL、失败原因、raw net / season excess / concentration
- 最近计划订单和 fill status tail
- 手动点击 `刷新 Alpaca Snapshot` 后,只调用 Alpaca paper 的 read endpoints,显示 account / recent orders / recent fills

Monitor 是执行状态看板,不是策略证明。策略是否值得跑仍然以 `signal_lab` / `intraday_seasonality_backtest` 的证伪关卡为准。

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

### `forward_paper.py`

前向 paper runner。它会拉 Polygon 30m 数据,用最近 lookback 天生成“下一交易日 13 个 slot”的订单计划,并写出 CSV 日志。默认是 shadow 模式,只记录订单意图,不更新 broker 账户。

```bash
python forward_paper.py --mode shadow --out paper_logs
```

本地 paper 执行:

```bash
python forward_paper.py --mode paper --notional 1000 --require-pass --out paper_logs
```

提交到 Alpaca paper:

```bash
$env:ALPACA_API_KEY="..."
$env:ALPACA_SECRET_KEY="..."
python forward_paper.py --broker alpaca-paper --mode paper --notional 1000 --require-pass --out paper_logs
```

输出文件:

- `paper_plan.csv`:每个 slot 的 pick 与历史 score
- `paper_orders.csv`:订单意图
- `paper_fills.csv`:本地 paper 成交
- `paper_summary.csv`:账户摘要
- `paper_research.csv`:研究闸门指标与 PASS/FAIL 原因

`--require-pass` 会在研究闸门 FAIL 时跳过执行,只保留计划和诊断。建议先 shadow 跑几周,确认信号、滑点、集中度和日志流程都稳定后,再考虑连接 broker paper API。

Alpaca paper adapter 也支持读取账户状态:

```python
from paper_broker import AlpacaPaperBroker

b = AlpacaPaperBroker()
print(b.get_account())
print(b.get_positions())
print(b.get_orders())
print(b.get_fills())
```

API key 只放环境变量或本地 git-ignored 配置,不要提交。即使是 paper key,泄露后也建议在 Alpaca dashboard 里 regenerate。

### `broker_benchmark.py`

Broker API 延迟/稳定性小基准。默认只测 Alpaca paper 的 read endpoints,不下单:

```bash
python broker_benchmark.py --broker alpaca-paper --n 10 --out broker_benchmark_logs
```

输出:

- `broker_latency_raw.csv`:每次调用的 latency/error
- `broker_latency_summary.csv`:每个 endpoint 的 p50/p95/max/error_rate

只有显式打开时才会发一笔 Alpaca paper 测试单并尝试取消:

```bash
python broker_benchmark.py --broker alpaca-paper --submit-test-order --symbol AAPL --notional 1
```

读法:对这个 repo 当前策略,一天最多十几笔订单,吞吐不是瓶颈;更该看 p95 latency、error_rate、以及 account/orders/fills 回读是否稳定。

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
