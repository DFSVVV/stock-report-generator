"""Report generator for stock analysis."""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np

from ..excel.models import StockDataBundle
from ..lstm import StockInference, calculate_bollinger_bands, calculate_macd, calculate_rsi, calculate_sma
from .formatters import format_date, format_percentage, format_price, format_volume
from .templates import REPORT_TEMPLATES, get_template


@dataclass
class AnalysisContext:
    """Context data for report generation."""

    stock_code: str
    stock_name: str
    latest_close: float
    sma_5: float
    sma_10: float
    sma_20: float
    rsi: float
    macd_dif: float
    macd_dea: float
    macd_hist: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    latest_turnover: float
    avg_turnover: float
    period_return: float
    period_high: float
    period_low: float
    predicted_return: float = 0.0
    confidence: float = 0.0
    predicted_trend: str = "STABLE"


class ReportGenerator:
    """Generator for stock analysis reports."""

    def __init__(self, inference: Optional[StockInference] = None):
        """Initialize report generator.

        Args:
            inference: Optional inference engine for predictions
        """
        self.inference = inference

    def generate(
        self,
        bundle: StockDataBundle,
        stock_name: Optional[str] = None,
        include_prediction: bool = True,
        period_days: int = 30,
    ) -> str:
        """Generate full report from StockDataBundle.

        Args:
            bundle: Stock data bundle
            stock_name: Optional stock name (if not provided, use stock_code)
            include_prediction: Whether to include LSTM prediction
            period_days: Number of days for period analysis

        Returns:
            Generated report string
        """
        context = self._build_context(bundle, stock_name, include_prediction, period_days)
        return self._render_report(context)

    def _build_context(
        self,
        bundle: StockDataBundle,
        stock_name: Optional[str],
        include_prediction: bool,
        period_days: int,
    ) -> AnalysisContext:
        """Compute all indicators and build analysis context."""
        closes = [d.close for d in bundle.data]
        volumes = [d.volume for d in bundle.data]
        turnover_rates = [d.turnover_rate for d in bundle.data]

        latest_close = closes[-1]

        # Calculate technical indicators
        sma_5_list = calculate_sma(closes, 5)
        sma_10_list = calculate_sma(closes, 10)
        sma_20_list = calculate_sma(closes, 20)

        sma_5 = sma_5_list[-1] if not np.isnan(sma_5_list[-1]) else latest_close
        sma_10 = sma_10_list[-1] if not np.isnan(sma_10_list[-1]) else latest_close
        sma_20 = sma_20_list[-1] if not np.isnan(sma_20_list[-1]) else latest_close

        rsi_list = calculate_rsi(closes, 14)
        rsi = rsi_list[-1] if not np.isnan(rsi_list[-1]) else 50.0

        macd_dif_list, macd_dea_list, macd_hist_list = calculate_macd(closes)
        macd_dif = macd_dif_list[-1] if not np.isnan(macd_dif_list[-1]) else 0.0
        macd_dea = macd_dea_list[-1] if not np.isnan(macd_dea_list[-1]) else 0.0
        macd_hist = macd_hist_list[-1] if not np.isnan(macd_hist_list[-1]) else 0.0

        bb_upper_list, bb_middle_list, bb_lower_list = calculate_bollinger_bands(closes, 20)
        bb_upper = bb_upper_list[-1] if not np.isnan(bb_upper_list[-1]) else latest_close * 1.1
        bb_middle = bb_middle_list[-1] if not np.isnan(bb_middle_list[-1]) else latest_close
        bb_lower = bb_lower_list[-1] if not np.isnan(bb_lower_list[-1]) else latest_close * 0.9

        # Period return
        if len(closes) >= period_days:
            period_start_close = closes[-period_days]
        else:
            period_start_close = closes[0]

        period_return = ((latest_close - period_start_close) / period_start_close * 100) if period_start_close > 0 else 0.0

        # Period high/low
        period_data = bundle.data[-period_days:] if len(bundle.data) >= period_days else bundle.data
        period_high = max(d.high for d in period_data)
        period_low = min(d.low for d in period_data)

        # Prediction
        predicted_return = 0.0
        confidence = 0.0
        predicted_trend = "STABLE"

        if include_prediction and self.inference:
            try:
                result = self.inference.predict(bundle)
                predicted_return = result.next_day_return
                confidence = result.confidence
                predicted_trend = result.trend
            except Exception:
                # If prediction fails, continue without it
                pass

        return AnalysisContext(
            stock_code=bundle.stock_code,
            stock_name=stock_name or bundle.stock_code,
            latest_close=latest_close,
            sma_5=sma_5,
            sma_10=sma_10,
            sma_20=sma_20,
            rsi=rsi,
            macd_dif=macd_dif,
            macd_dea=macd_dea,
            macd_hist=macd_hist,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            latest_turnover=turnover_rates[-1] if turnover_rates else 0.0,
            avg_turnover=sum(turnover_rates) / len(turnover_rates) if turnover_rates else 0.0,
            period_return=period_return,
            period_high=period_high,
            period_low=period_low,
            predicted_return=predicted_return,
            confidence=confidence,
            predicted_trend=predicted_trend,
        )

    def _render_report(self, context: AnalysisContext) -> str:
        """Render full report from context."""
        lines = []

        # Header
        header = REPORT_TEMPLATES["header"].format(
            stock_code=context.stock_code,
            stock_name=context.stock_name,
            report_date=format_date(date.today()),
        )
        lines.append(header)

        # Summary section
        bundle_dates = [d.trade_date for d in getattr(self, "_bundle", []).data] if hasattr(self, "_bundle") else []
        if not bundle_dates:
            bundle_dates = [date.today()]

        summary = REPORT_TEMPLATES["summary"].format(
            stock_code=context.stock_code,
            stock_name=context.stock_name,
            start_date=format_date(bundle_dates[0]),
            end_date=format_date(bundle_dates[-1]),
            num_trading_days=len(bundle_dates),
            latest_close=format_price(context.latest_close),
            period="近30日" if True else "近N日",
            period_return=format_percentage(context.period_return, include_sign=True),
            period_high=format_price(context.period_high),
            period_low=format_price(context.period_low),
            latest_turnover=format_percentage(context.latest_turnover, include_sign=False),
            avg_turnover=format_percentage(context.avg_turnover, include_sign=False),
            price_above_ma5=context.latest_close > context.sma_5,
            price_above_ma10=context.latest_close > context.sma_10,
            price_above_ma20=context.latest_close > context.sma_20,
        )
        lines.append(summary)

        # Technical section
        rsi_status = "超买" if context.rsi > 70 else "超卖" if context.rsi < 30 else "正常"
        macd_signal = (
            "金叉(看涨)" if context.macd_dif > context.macd_dea else "死叉(看跌)"
        )
        bb_position = (
            "接近上轨(超买)" if context.latest_close > context.bb_upper * 0.98
            else "接近下轨(超卖)" if context.latest_close < context.bb_lower * 1.02
            else "中轨附近"
        )

        technical = REPORT_TEMPLATES["technical"].format(
            sma_5=context.sma_5,
            sma_10=context.sma_10,
            sma_20=context.sma_20,
            price_above_ma5="高于" if context.latest_close > context.sma_5 else "低于",
            price_above_ma10="高于" if context.latest_close > context.sma_10 else "低于",
            price_above_ma20="高于" if context.latest_close > context.sma_20 else "低于",
            rsi=context.rsi,
            rsi_status=rsi_status,
            macd_dif=context.macd_dif,
            macd_dea=context.macd_dea,
            macd_hist=context.macd_hist,
            macd_signal=macd_signal,
            bb_upper=context.bb_upper,
            bb_middle=context.bb_middle,
            bb_lower=context.bb_lower,
            bb_position=bb_position,
        )
        lines.append(technical)

        # Prediction section
        suggestion = self._get_investment_suggestion(context)
        prediction = REPORT_TEMPLATES["prediction"].format(
            predicted_return=context.predicted_return,
            confidence=context.confidence,
            predicted_trend=context.predicted_trend,
            investment_suggestion=suggestion,
        )
        lines.append(prediction)

        # Risk section
        risk = self._get_risk_assessment(context)
        risk_text = REPORT_TEMPLATES["risk"].format(
            volatility=risk["volatility"],
            volume_alert=risk["volume_alert"],
            reversal_signal=risk["reversal_signal"],
        )
        lines.append(risk_text)

        return "\n".join(lines)

    def _get_investment_suggestion(self, context: AnalysisContext) -> str:
        """Generate investment suggestion based on analysis."""
        score = 0
        reasons = []

        # Trend analysis
        if context.predicted_trend == "UP":
            score += 2
            reasons.append("模型预测上涨")
        elif context.predicted_trend == "DOWN":
            score -= 2
            reasons.append("模型预测下跌")

        # RSI analysis
        if context.rsi < 30:
            score += 1
            reasons.append("RSI超卖，可能反弹")
        elif context.rsi > 70:
            score -= 1
            reasons.append("RSI超买，注意回调风险")

        # MACD analysis
        if context.macd_dif > context.macd_dea and context.macd_hist > 0:
            score += 1
            reasons.append("MACD金叉")
        elif context.macd_dif < context.macd_dea and context.macd_hist < 0:
            score -= 1
            reasons.append("MACD死叉")

        # MA analysis
        ma_signals = 0
        if context.latest_close > context.sma_5:
            ma_signals += 1
        if context.latest_close > context.sma_10:
            ma_signals += 1
        if context.latest_close > context.sma_20:
            ma_signals += 1

        if ma_signals >= 2:
            score += 1
            reasons.append("价格位于均线上方")
        elif ma_signals <= 1:
            score -= 1
            reasons.append("价格位于均线下方")

        # Generate suggestion
        if score >= 3:
            return f"建议关注/轻仓参与 ({', '.join(reasons)})"
        elif score >= 1:
            return f"谨慎观望 ({', '.join(reasons)})"
        elif score <= -3:
            return f"建议回避 ({', '.join(reasons)})"
        else:
            return f"中性观望 ({', '.join(reasons)})"

    def _get_risk_assessment(self, context: AnalysisContext) -> dict:
        """Assess various risk factors."""
        # Volatility risk
        price_range = context.period_high - context.period_low
        volatility_pct = (price_range / context.latest_close * 100) if context.latest_close > 0 else 0

        if volatility_pct > 20:
            volatility = f"高波动 (振幅{volatility_pct:.1f}%)"
        elif volatility_pct > 10:
            volatility = f"中等波动 (振幅{volatility_pct:.1f}%)"
        else:
            volatility = f"低波动 (振幅{volatility_pct:.1f}%)"

        # Volume risk
        avg_vol_ma = getattr(self, "_recent_vol_ma", 0)
        if avg_vol_ma > 0:
            vol_ratio = context.latest_turnover / (avg_vol_ma / 1e6) if avg_vol_ma > 0 else 1.0
            if vol_ratio > 2.0:
                volume_alert = f"成交量异常放大 ({vol_ratio:.1f}倍)"
            elif vol_ratio < 0.3:
                volume_alert = f"成交量异常萎缩 ({vol_ratio:.1f}倍)"
            else:
                volume_alert = "成交量正常"
        else:
            volume_alert = "成交量正常"

        # Reversal signals
        reversal_signals = []
        if context.rsi > 70:
            reversal_signals.append("RSI超买")
        if context.rsi < 30:
            reversal_signals.append("RSI超卖")
        if context.macd_hist < 0 and context.macd_dif < context.macd_dea:
            reversal_signals.append("MACD顶背离")
        if context.latest_close < context.bb_lower:
            reversal_signals.append("价格跌破布林下轨")

        reversal_signal = "无明显信号" if not reversal_signals else ", ".join(reversal_signals)

        return {
            "volatility": volatility,
            "volume_alert": volume_alert,
            "reversal_signal": reversal_signal,
        }
