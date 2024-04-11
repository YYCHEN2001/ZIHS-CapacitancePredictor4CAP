import pandas as pd

# 加载数据
data_path = '../../data/raw/cleaned_data_after_mvp.csv'
data = pd.read_csv(data_path)

# 进行分布统计
stats_report = data.describe()

# 计算值域并添加到报告中
stats_report.loc['range'] = stats_report.loc['max'] - stats_report.loc['min']

# 转换为Markdown格式
stats_md = stats_report.to_markdown()

# 保存报告为Markdown格式
report_path = '../../reports/data/feature_distribution_stats_after_mvp.md'
with open(report_path, 'w') as f:
    f.write(stats_md)