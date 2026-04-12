"""End-to-end pipeline for stock report generation."""

from datetime import date
from pathlib import Path
from typing import Optional

from ..excel import ExcelReader, StockDataBundle
from ..lstm import StockInference
from ..report import ReportGenerator


class StockReportPipeline:
    """End-to-end pipeline for generating stock analysis reports.

    Supports two data sources:
    1. Excel file: Local Excel files with stock data
    2. Online: Real-time data via akshare

    Usage:
        # From Excel file
        pipeline = StockReportPipeline()
        report = pipeline.generate_from_excel('data/stock.xlsx', stock_name='平安银行')

        # From akshare (real-time data)
        pipeline = StockReportPipeline()
        report = pipeline.generate_from_online('000001', stock_name='平安银行',
                                               start_date='20240101', end_date='20240412')
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        sequence_length: int = 60,
    ):
        """Initialize pipeline.

        Args:
            model_path: Path to trained LSTM model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            sequence_length: Sequence length for LSTM
        """
        self.model_path = model_path
        self.device = device
        self.sequence_length = sequence_length

        # Initialize components
        self.inference: Optional[StockInference] = None
        if model_path and Path(model_path).exists():
            self.inference = StockInference(
                model_path=model_path,
                device=device,
                sequence_length=sequence_length,
            )

        self.generator = ReportGenerator(inference=self.inference)

    def generate_from_excel(
        self,
        excel_path: str,
        stock_name: Optional[str] = None,
        include_prediction: bool = True,
    ) -> str:
        """Generate report from Excel file.

        Args:
            excel_path: Path to Excel file with stock data
            stock_name: Optional stock name override
            include_prediction: Whether to include LSTM prediction

        Returns:
            Generated report string
        """
        reader = ExcelReader(excel_path)
        result = reader.validate()

        if not result.valid:
            raise ValueError(f"Invalid Excel file: {result.errors}")

        bundle = reader.read()
        return self.generate_from_bundle(bundle, stock_name, include_prediction)

    def generate_from_online(
        self,
        stock_code: str,
        stock_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_prediction: bool = True,
        data_source: str = "sina",
    ) -> str:
        """Generate report from real-time data.

        Args:
            stock_code: Stock code (e.g., '000001')
            stock_name: Optional stock name
            start_date: Start date in YYYYMMDD format (default: 1 year ago)
            end_date: End date in YYYYMMDD format (default: today)
            include_prediction: Whether to include LSTM prediction
            data_source: Data source to use: 'sina' (recommended), 'eastmoney',
                        or 'akshare'

        Returns:
            Generated report string

        Raises:
            ImportError: If akshare is not installed (when data_source='akshare')
            ConnectionError: If fails to fetch data from network
        """
        if data_source == "sina":
            from ..data_source import SinaDataSource
            source = SinaDataSource()
        elif data_source == "eastmoney":
            from ..data_source import EastMoneyDataSource
            source = EastMoneyDataSource()
        elif data_source == "akshare":
            from ..data_source import AkshareDataSource
            source = AkshareDataSource()
        else:
            raise ValueError(f"Unknown data source: {data_source}. Use 'sina', 'eastmoney', or 'akshare'")

        # Default dates
        if end_date is None:
            end_date = date.today().strftime("%Y%m%d")
        if start_date is None:
            # Default to 1 year ago
            from datetime import timedelta
            start_date_dt = date.today() - timedelta(days=365)
            start_date = start_date_dt.strftime("%Y%m%d")

        bundle = source.fetch_with_name(
            stock_code=stock_code,
            stock_name=stock_name,
            start_date=start_date,
            end_date=end_date,
        )

        return self.generate_from_bundle(bundle, stock_name, include_prediction)

    def generate_from_bundle(
        self,
        bundle: StockDataBundle,
        stock_name: Optional[str] = None,
        include_prediction: bool = True,
    ) -> str:
        """Generate report from StockDataBundle.

        Args:
            bundle: Stock data bundle
            stock_name: Optional stock name
            include_prediction: Whether to include LSTM prediction

        Returns:
            Generated report string
        """
        if include_prediction and not self.inference:
            raise ValueError(
                "Prediction requested but no model loaded. "
                "Provide model_path during initialization."
            )

        # Store bundle reference for date formatting in generator
        self.generator._bundle = bundle

        report = self.generator.generate(
            bundle=bundle,
            stock_name=stock_name,
            include_prediction=include_prediction,
        )

        return report

    def export_report(
        self,
        report: str,
        output_path: str,
    ) -> None:
        """Export report to text file.

        Args:
            report: Report content
            output_path: Output file path
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
