# Stock Report Generator

基于深度学习的股票数据分析报告自动生成系统

## 技术栈

- **文本生成**: LSTM + BERT
- **异常检测**: IsolationForest
- **文档处理**: python-docx, openpyxl
- **图表生成**: matplotlib, seaborn
- **CI/CD**: GitHub Actions

## 项目结构

```
stock-report-generator/
├── .github/workflows/   # CI/CD 配置
├── src/                 # 源代码
├── tests/               # 测试
├── docs/                # 文档
├── requirements.txt     # 依赖
└── pyproject.toml       # 项目配置
```

## 快速开始

```bash
pip install -r requirements.txt
python -m pytest tests/
```

## 开发

```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 运行测试
python -m pytest tests/ -v
```
