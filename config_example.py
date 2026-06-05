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
