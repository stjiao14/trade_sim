HOLDINGS = dict(
    googl_shares=0,            # 直接持有 GOOGL 股数
    rsu_unvested_shares=0,     # 未归属 RSU 股数(经济敞口)
    spy_usd=0.0, qqq_usd=0.0, tqqq_usd=0.0,   # 各 ETF 市值(美元)
)

SALARY_USD = 0.0              # 年薪(税前);设 0 可关掉人力资本那行
HC_YEARS   = 0                # 计入几年人力资本(示意,例如 5);0 = 不计

# ETF 内 Alphabet(GOOGL+GOOG 合计)权重的手动兜底值(自动抓取失败时用)。
# 核对来源:QQQ -> Invesco 持仓页;SPY -> SSGA SPDR 持仓页。
FALLBACK_WEIGHTS = dict(spy_alphabet=0.038, qqq_alphabet=0.050)

# 可选:本地真实持仓 CSV 路径。真实路径请放 config_local.py,不要提交。
HOLDINGS_CSV = ""

# 可选:Web UI / CLI 共用的本地 API 配置。真实 key 请放 config_local.py,不要提交。
ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
POLYGON_API_KEY = ""

# RSU 归属后 HOLD vs SELL 分散模拟器用的 dummy 配置。
# 真实数字请放 config_local.py 的 VEST_HOLDINGS 里,不要提交。
VEST_HOLDINGS = dict(
    goog_price=250.0,           # 当前 GOOG/GOOGL 价(仅用于 RSU 股数换算)
    rsu_unvested_usd=100_000.0, # 未归属 RSU 现值(美元);dummy 示例,非真实金额
    liquid_goog_usd=20_000.0,   # 流动账户里的 GOOG 等价敞口;dummy 示例
    liquid_basket_usd=180_000.0,# 流动账户其余(视作分散篮子);dummy 示例
)
VEST_MONTHS = list(range(3, 49, 3))
BASKET_PROXY = "VT"
N_PATHS = 20000

# Paper trading / forward-test dummy 配置。真实 key 用环境变量,不要写进 config。
PAPER_TRADING = dict(
    starting_cash=100_000.0,
    max_order_notional=5_000.0,
    max_symbol_notional=20_000.0,
    max_gross_notional=100_000.0,
    allow_short=False,
    blocked_symbols=[],
    slippage_bps=1.0,
    commission_bps=0.0,
    alpaca_base_url="https://paper-api.alpaca.markets",
    log_dir="paper_logs",
)
