"""PyQt5 GUI application for Stock Report Generator."""

import os
import sys
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QLineEdit,
    QTextEdit,
    QGroupBox,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont


class ReportGeneratorWorker(QThread):
    """Worker thread for generating reports."""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode  # 'excel' or 'online'
        self.excel_path = None
        self.stock_code = None
        self.stock_name = None
        self.template_path = None
        self.use_template = False

    def run(self):
        try:
            self.progress.emit(10, "初始化...")
            from src.pipeline import StockReportPipeline

            # Load model if exists
            model_path = "models/stock_lstm.pt"
            if os.path.exists(model_path):
                pipeline = StockReportPipeline(model_path=model_path)
            else:
                pipeline = StockReportPipeline()

            self.progress.emit(30, "读取数据...")

            if self.mode == "excel":
                if not self.excel_path:
                    raise ValueError("请选择Excel文件")
                bundle = None
                stock_name = self.stock_name.strip() if self.stock_name else None
            else:
                if not self.stock_code:
                    raise ValueError("请输入股票代码")
                stock_code = self.stock_code.strip()
                # Keep user input name, or None to auto-detect via Tencent API
                stock_name = self.stock_name.strip() if self.stock_name else None

                self.progress.emit(40, f"正在获取 {stock_code} 数据...")
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = f"{int(end_date[:4]) - 1}{end_date[4:]}"

                try:
                    from src.data_source import SinaDataSource

                    source = SinaDataSource()
                    bundle = source.fetch_with_name(
                        stock_code=stock_code,
                        stock_name=stock_name,
                        start_date=start_date,
                        end_date=end_date,
                    )
                    # fetch_with_name returns (bundle, actual_name)
                    if isinstance(bundle, tuple):
                        bundle, actual_name = bundle
                        stock_name = actual_name
                    else:
                        stock_name = stock_name or stock_code
                except ImportError:
                    raise ImportError("请先安装 akshare: pip install akshare")
                except Exception as e:
                    raise Exception(f"获取数据失败: {str(e)}")

            self.progress.emit(60, "生成分析报告...")

            if self.use_template and self.template_path:
                # Generate report from template
                from src.template import DocxTemplateProcessor

                processor = DocxTemplateProcessor()
                output_path = self.excel_path.replace(".xlsx", "_报告.docx") if self.excel_path else "output_report.docx"
                processor.process(
                    template_path=self.template_path,
                    bundle=bundle,
                    output_path=output_path,
                    stock_name=stock_name,
                )
                report = f"✅ 报告已生成！\n\n文件位置: {output_path}\n\n使用模板: {self.template_path}"
            else:
                # Generate text report
                if self.mode == "excel":
                    report = pipeline.generate_from_excel(
                        self.excel_path,
                        stock_name=stock_name,
                        include_prediction=True,
                    )
                else:
                    report = pipeline.generate_from_bundle(
                        bundle,
                        stock_name=stock_name,
                        include_prediction=True,
                    )

            self.progress.emit(100, "完成!")
            self.finished.emit(report)

        except ImportError as e:
            self.error.emit(str(e))
        except Exception as e:
            self.error.emit(str(e))


class StockReportGUI(QMainWindow):
    """Main GUI window for Stock Report Generator."""

    def __init__(self):
        super().__init__()
        self.current_report = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("股票数据分析报告生成器")
        self.setMinimumSize(900, 750)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # Title
        title_label = QLabel("股票数据分析报告生成器")
        title_label.setFont(QFont("Microsoft YaHei", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Subtitle
        subtitle = QLabel("基于深度学习的股票数据分析报告自动生成系统")
        subtitle.setFont(QFont("Microsoft YaHei", 9))
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        main_layout.addSpacing(10)

        # ============ Mode Selection Group ============
        mode_group = QGroupBox("选择分析模式")
        mode_layout = QVBoxLayout()
        mode_group.setLayout(mode_layout)
        main_layout.addWidget(mode_group)

        # ========== Excel Mode ==========
        self.excel_btn = QPushButton("📁 方式一：通过 Excel 文件分析")
        self.excel_btn.setFont(QFont("Microsoft YaHei", 11))
        self.excel_btn.setMinimumHeight(45)
        self.excel_btn.clicked.connect(self.on_excel_mode)
        mode_layout.addWidget(self.excel_btn)

        # Excel file path
        excel_path_layout = QHBoxLayout()
        self.excel_path_label = QLabel("未选择文件")
        self.excel_path_label.setStyleSheet("color: gray;")
        self.excel_path_label.setWordWrap(True)
        self.excel_browse_btn = QPushButton("浏览数据...")
        self.excel_browse_btn.clicked.connect(self.browse_excel_file)
        excel_path_layout.addWidget(QLabel("数据文件:"))
        excel_path_layout.addWidget(self.excel_path_label, 1)
        excel_path_layout.addWidget(self.excel_browse_btn)
        mode_layout.addLayout(excel_path_layout)

        # Template file path
        template_path_layout = QHBoxLayout()
        self.template_path_label = QLabel("使用默认模板")
        self.template_path_label.setStyleSheet("color: gray;")
        self.template_path_label.setWordWrap(True)
        self.template_browse_btn = QPushButton("浏览模板...")
        self.template_browse_btn.clicked.connect(self.browse_template_file)
        template_path_layout.addWidget(QLabel("模板文件:"))
        template_path_layout.addWidget(self.template_path_label, 1)
        template_path_layout.addWidget(self.template_browse_btn)
        mode_layout.addLayout(template_path_layout)

        # Template checkbox
        self.use_template_cb = QCheckBox("使用模板生成 Word 报告 (docx)")
        self.use_template_cb.setChecked(False)
        self.use_template_cb.toggled.connect(self.on_template_toggled)
        mode_layout.addWidget(self.use_template_cb)

        # Divider
        divider_layout = QHBoxLayout()
        divider = QLabel("─" * 40)
        divider.setAlignment(Qt.AlignCenter)
        divider_layout.addWidget(divider)
        mode_layout.addLayout(divider_layout)

        # ========== Online Mode ==========
        self.online_btn = QPushButton("🌐 方式二：输入股票代码在线分析")
        self.online_btn.setFont(QFont("Microsoft YaHei", 11))
        self.online_btn.setMinimumHeight(45)
        self.online_btn.clicked.connect(self.on_online_mode)
        mode_layout.addWidget(self.online_btn)

        # Stock code input
        stock_input_layout = QHBoxLayout()
        stock_input_layout.addWidget(QLabel("股票代码:"))
        self.stock_code_input = QLineEdit()
        self.stock_code_input.setPlaceholderText("例如: 000001")
        stock_input_layout.addWidget(self.stock_code_input, 1)

        stock_input_layout.addWidget(QLabel("名称(可选):"))
        self.stock_name_input = QLineEdit()
        self.stock_name_input.setPlaceholderText("例如: 平安银行")
        stock_input_layout.addWidget(self.stock_name_input, 1)

        mode_layout.addLayout(stock_input_layout)

        # Tips
        tips_label = QLabel("提示: 股票代码如 000001(平安银行), 600519(贵州茅台), 000002(万科A)")
        tips_label.setStyleSheet("color: #666; font-size: 10px;")
        tips_label.setWordWrap(True)
        mode_layout.addWidget(tips_label)

        main_layout.addSpacing(10)

        # ============ Report Display Group ============
        report_group = QGroupBox("生成的报告")
        report_layout = QVBoxLayout()
        report_group.setLayout(report_layout)
        main_layout.addWidget(report_group, 1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        report_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #0066cc;")
        report_layout.addWidget(self.status_label)

        # Report text area with scroll
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Consolas", 10))
        scroll_area.setWidget(self.report_text)
        report_layout.addWidget(scroll_area, 1)

        # ============ Bottom Buttons ============
        bottom_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 保存报告")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_report)
        bottom_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("🗑️ 清空")
        self.clear_btn.clicked.connect(self.clear_report)
        bottom_layout.addWidget(self.clear_btn)

        main_layout.addLayout(bottom_layout)

    def on_template_toggled(self, checked):
        """Handle template checkbox toggle."""
        self.template_browse_btn.setEnabled(checked)
        if checked:
            self.template_path_label.setText("请选择模板文件...")
            self.template_path_label.setStyleSheet("color: orange;")
        else:
            self.template_path_label.setText("使用默认模板")
            self.template_path_label.setStyleSheet("color: gray;")

    def browse_excel_file(self):
        """Open file dialog to select Excel file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Excel 数据文件", "", "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.excel_path_label.setText(file_path)
            self.excel_path_label.setStyleSheet("color: black;")

    def browse_template_file(self):
        """Open file dialog to select template file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择报告模板", "", "Word Files (*.docx);;All Files (*)"
        )
        if file_path:
            self.template_path_label.setText(file_path)
            self.template_path_label.setStyleSheet("color: green;")
            self.use_template_cb.setChecked(True)

    def on_excel_mode(self):
        """Handle Excel mode button click."""
        excel_path = self.excel_path_label.text()
        if excel_path == "未选择文件" or not excel_path:
            QMessageBox.warning(self, "提示", "请先选择 Excel 数据文件")
            return

        stock_name = self.stock_name_input.text().strip()
        use_template = self.use_template_cb.isChecked()
        template_path = self.template_path_label.text() if use_template else None

        if use_template and (template_path == "请选择模板文件..." or not template_path):
            QMessageBox.warning(self, "提示", "请先选择模板文件")
            return

        self.run_generation(
            mode="excel",
            excel_path=excel_path,
            stock_name=stock_name,
            use_template=use_template,
            template_path=template_path,
        )

    def on_online_mode(self):
        """Handle online mode button click."""
        stock_code = self.stock_code_input.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "提示", "请输入股票代码")
            return

        stock_name = self.stock_name_input.text().strip()
        use_template = self.use_template_cb.isChecked()
        template_path = self.template_path_label.text() if use_template else None

        self.run_generation(
            mode="online",
            stock_code=stock_code,
            stock_name=stock_name,
            use_template=use_template,
            template_path=template_path,
        )

    def run_generation(self, mode, **kwargs):
        """Run report generation in background."""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "提示", "正在生成中，请稍候...")
            return

        self.excel_btn.setEnabled(False)
        self.online_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.report_text.clear()
        self.status_label.setText("准备中...")

        self.worker = ReportGeneratorWorker(mode)
        self.worker.excel_path = kwargs.get("excel_path")
        self.worker.stock_code = kwargs.get("stock_code")
        self.worker.stock_name = kwargs.get("stock_name")
        self.worker.use_template = kwargs.get("use_template", False)
        self.worker.template_path = kwargs.get("template_path")

        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_progress(self, value, message):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def on_finished(self, report):
        """Handle generation finished."""
        self.current_report = report
        self.report_text.setPlainText(report)
        self.status_label.setText("✅ 报告生成完成!")
        self.save_btn.setEnabled(True)
        self.excel_btn.setEnabled(True)
        self.online_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    def on_error(self, error_message):
        """Handle generation error."""
        self.status_label.setText(f"❌ 错误: {error_message}")
        self.report_text.setPlainText(f"生成失败:\n\n{error_message}")
        self.excel_btn.setEnabled(True)
        self.online_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        if "akshare" in error_message.lower():
            QMessageBox.critical(
                self,
                "缺少依赖",
                f"{error_message}\n\n请在命令行运行: pip install akshare",
            )

    def save_report(self):
        """Save report to file."""
        if not self.current_report:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存报告", "", "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(self.current_report)
                QMessageBox.information(self, "成功", f"报告已保存到:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败:\n{str(e)}")

    def clear_report(self):
        """Clear the current report."""
        self.current_report = None
        self.report_text.clear()
        self.status_label.clear()
        self.save_btn.setEnabled(False)
        self.excel_path_label.setText("未选择文件")
        self.excel_path_label.setStyleSheet("color: gray;")
        self.template_path_label.setText("使用默认模板")
        self.template_path_label.setStyleSheet("color: gray;")
        self.use_template_cb.setChecked(False)
        self.stock_code_input.clear()
        self.stock_name_input.clear()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)

    # Set stylesheet
    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #f5f5f5;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
        QPushButton:disabled {
            background-color: #ccc;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        QLineEdit {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        QLineEdit:focus {
            border: 1px solid #0078d4;
        }
        QProgressBar {
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: center;
        }
        QCheckBox {
            padding: 5px 0;
        }
        """
    )

    window = StockReportGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
