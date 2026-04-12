"""Report templates for stock analysis."""

REPORT_TEMPLATES = {
    "header": """
============================================================
【{stock_code} {stock_name} 技术分析报告】
============================================================
生成日期: {report_date}
""",
    "summary": """
一、基本信息
----------------------------------------
  股票代码: {stock_code}
  股票名称: {stock_name}
  数据区间: {start_date} 至 {end_date}
  交易日数量: {num_trading_days}天

二、近期走势
----------------------------------------
  最新收盘价: {latest_close}元
  {period}涨跌幅: {period_return}%
  期间最高价: {period_high}元
  期间最低价: {period_low}元
  最新换手率: {latest_turnover}%
  平均换手率: {avg_turnover}%
""",
    "technical": """
三、技术指标分析
----------------------------------------
  1. 移动平均线
     MA5:  {sma_5:.2f}元 (当前价格{price_above_ma5}MA5)
     MA10: {sma_10:.2f}元 (当前价格{price_above_ma10}MA10)
     MA20: {sma_20:.2f}元 (当前价格{price_above_ma20}MA20)

  2. RSI指标
     RSI(14): {rsi:.1f}
     市场状态: {rsi_status} (超买>70, 超卖<30, 正常)

  3. MACD指标
     DIF: {macd_dif:.4f}
     DEA: {macd_dea:.4f}
     MACD柱: {macd_hist:.4f}
     信号: {macd_signal}

  4. 布林带
     上轨: {bb_upper:.2f}元
     中轨: {bb_middle:.2f}元
     下轨: {bb_lower:.2f}元
     位置: {bb_position}
""",
    "prediction": """
四、LSTM模型预测
----------------------------------------
  预测下一个交易日涨跌幅: {predicted_return:.2f}%
  预测置信度: {confidence:.1%}
  趋势判断: {predicted_trend}

五、投资建议
----------------------------------------
  {investment_suggestion}
""",
    "risk": """
六、风险提示
----------------------------------------
  1. 波动性风险: {volatility}
  2. 成交量异常: {volume_alert}
  3. 趋势反转信号: {reversal_signal}

============================================================
""",
}


def get_template(name: str) -> str:
    """Get template by name.

    Args:
        name: Template name

    Returns:
        Template string

    Raises:
        KeyError: If template not found
    """
    if name not in REPORT_TEMPLATES:
        raise KeyError(f"Template '{name}' not found. Available: {list(REPORT_TEMPLATES.keys())}")
    return REPORT_TEMPLATES[name]
