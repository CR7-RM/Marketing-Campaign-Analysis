# Marketing Campaign Time Series Analysis

这是一个用于分析营销广告活动时间序列数据的 Python 项目。它能够处理广告数据，计算关键指标（如CTR），并进行趋势分析和未来预测。

## 📁 文件说明

| 文件名称 | 类型 | 描述 |
| :--- | :--- | :--- |
| Marketing Campaign Analysis.xlsx | 数据文件 | 包含广告活动的原始数据，如展示量、点击量、花费等。 |
| marketing_campaign_analysis.py | Python 脚本 | 主程序文件，用于数据清洗、分析和可视化。 |

## 🚀 功能特点

- **数据清洗与预处理**: 自动处理日期格式和缺失值。
- **核心指标计算**: 计算点击率（CTR）、单次点击成本（CPC）等。
- **时间序列分析**: 分析广告效果随时间的趋势和波动。
- **可视化图表**: 生成趋势图、周度对比图等，直观展示分析结果。
- **预测模型**: 使用移动平均法对未来一段时间的CTR进行预测。
- **渠道分析**: 对比不同广告渠道的效果表现。

## ⚙️ 使用方法

1. **环境准备**: 确保您的电脑已安装 Python，并通过 pip 安装所需的库：
bash

pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

复制
2. **运行脚本**: 将 `Marketing Campaign Analysis.xlsx` 文件放在与 Python 脚本相同的目录下，然后运行脚本：
bash

python marketing_campaign_analysis.py

复制
3. **查看结果**: 程序运行后，会在同一目录下生成分析图表（PNG格式）和结果文件（Excel/CSV格式）。

## 📊 输出结果

运行脚本后，将会生成以下文件：
- `广告趋势分析.png`: 展示CTR、展示量、点击量和CPC的趋势图。
- `周度分析.png`: 一周内各天平均CTR的对比图。
- `CTR预测结果.png`: 包含历史数据、测试集预测和未来预测的综合图表。
- `渠道分析.png` (如果数据包含渠道信息): 各广告渠道的CTR对比图。
- `CTR时间序列分析结果.xlsx`: 包含历史及预测CTR的数据表格。
- `分析统计汇总.csv`: 关键指标的统计汇总。

## ❓ 常见问题

- **确保Excel文件名称正确**: 脚本默认寻找名为 `Marketing Campaign Analysis.xlsx` 的文件，请保持文件名一致。
- **日期格式问题**: 如果遇到日期解析错误，请检查Excel文件中日期列的数据格式。

## 👥 贡献

欢迎提交 Issue 或 Pull Request 来改进这个项目。
