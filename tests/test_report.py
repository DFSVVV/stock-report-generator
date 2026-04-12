"""Tests for report generation module."""

from datetime import date

import pytest

from src.report import (
    ReportGenerator,
    AnalysisContext,
    format_price,
    format_percentage,
    format_volume,
    format_date,
)
from src.report.formatters import format_date_chinese


class TestFormatters:
    """Test formatting utilities."""

    def test_format_price(self) -> None:
        """Test price formatting."""
        assert format_price(10.0) == "10.00"
        assert format_price(10.123) == "10.12"
        assert format_price(10.126) == "10.13"

    def test_format_percentage(self) -> None:
        """Test percentage formatting."""
        assert format_percentage(10.5, include_sign=True) == "+10.50%"
        assert format_percentage(-5.0, include_sign=True) == "-5.00%"
        assert format_percentage(0.0, include_sign=True) == "0.00%"

        assert format_percentage(10.5, include_sign=False) == "10.50%"
        assert format_percentage(-5.0, include_sign=False) == "-5.00%"

    def test_format_volume(self) -> None:
        """Test volume formatting."""
        assert format_volume(1000) == "1000"
        assert format_volume(15000) == "2万"
        assert format_volume(1500000) == "150万"
        assert format_volume(150000000) == "1.50亿"

    def test_format_date(self) -> None:
        """Test date formatting."""
        d = date(2024, 1, 15)
        assert format_date(d) == "2024-01-15"

    def test_format_date_chinese(self) -> None:
        """Test Chinese date formatting."""
        d = date(2024, 1, 15)
        assert format_date_chinese(d) == "2024年01月15日"


class TestReportGenerator:
    """Test ReportGenerator."""

    @pytest.fixture
    def sample_context(self) -> AnalysisContext:
        """Create sample analysis context."""
        return AnalysisContext(
            stock_code="000001",
            stock_name="平安银行",
            latest_close=12.50,
            sma_5=12.30,
            sma_10=12.20,
            sma_20=12.00,
            rsi=55.0,
            macd_dif=0.15,
            macd_dea=0.10,
            macd_hist=0.10,
            bb_upper=13.00,
            bb_middle=12.50,
            bb_lower=11.50,
            latest_turnover=0.8,
            avg_turnover=0.6,
            period_return=5.2,
            period_high=13.00,
            period_low=11.50,
            predicted_return=1.5,
            confidence=0.75,
            predicted_trend="UP",
        )

    def test_get_investment_suggestion_uptrend(self, sample_context: AnalysisContext) -> None:
        """Test investment suggestion for uptrend."""
        generator = ReportGenerator()

        suggestion = generator._get_investment_suggestion(sample_context)

        assert "建议" in suggestion or "关注" in suggestion or "谨慎" in suggestion

    def test_get_investment_suggestion_downtrend(self, sample_context: AnalysisContext) -> None:
        """Test investment suggestion for downtrend."""
        # Create a fresh context to avoid mutation issues
        downtrend_context = AnalysisContext(
            stock_code="000001",
            stock_name="Test Stock",
            latest_close=12.50,
            sma_5=12.60,
            sma_10=12.70,
            sma_20=12.80,
            rsi=75.0,
            macd_dif=0.05,
            macd_dea=0.10,
            macd_hist=-0.10,
            bb_upper=13.50,
            bb_middle=12.50,
            bb_lower=11.50,
            latest_turnover=0.8,
            avg_turnover=0.6,
            period_return=-5.0,
            period_high=13.00,
            period_low=11.50,
            predicted_return=-3.0,
            confidence=0.75,
            predicted_trend="DOWN",
        )

        generator = ReportGenerator()

        suggestion = generator._get_investment_suggestion(downtrend_context)

        assert "回避" in suggestion or "建议" in suggestion or "谨慎" in suggestion or "中性" in suggestion

    def test_get_risk_assessment(self, sample_context: AnalysisContext) -> None:
        """Test risk assessment."""
        generator = ReportGenerator()

        risk = generator._get_risk_assessment(sample_context)

        assert "volatility" in risk
        assert "volume_alert" in risk
        assert "reversal_signal" in risk
        assert isinstance(risk["volatility"], str)
        assert isinstance(risk["volume_alert"], str)
        assert isinstance(risk["reversal_signal"], str)

    def test_get_risk_assessment_high_volatility(self, sample_context: AnalysisContext) -> None:
        """Test risk assessment with high volatility."""
        sample_context.period_high = 15.00
        sample_context.period_low = 10.00

        generator = ReportGenerator()

        risk = generator._get_risk_assessment(sample_context)

        assert "高波动" in risk["volatility"]
