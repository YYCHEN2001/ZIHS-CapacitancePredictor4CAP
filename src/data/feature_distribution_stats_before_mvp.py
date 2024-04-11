import pandas as pd


def generate_feature_distribution_stats_md(data_path, report_path):
    # 加载数据
    data = pd.read_csv(data_path)

    # 对所有数值型特征进行分布统计
    stats_report = data.describe()

    # 计算值域并添加到报告中
    stats_report.loc['range'] = stats_report.loc['max'] - stats_report.loc['min']

    # 转换为Markdown格式
    stats_md = stats_report.to_markdown()

    # 保存报告为Markdown格式
    with open(report_path, 'w') as f:
        f.write(stats_md)


if __name__ == "__main__":
    data_path = '../../data/raw/cleaned_data.csv'  # 替换为您的数据文件路径
    report_path = '../../reports/data/feature_distribution_stats_before_mvp.md'  # 替换为您的报告保存路径
    generate_feature_distribution_stats_md(data_path, report_path)
    print("Feature distribution statistics report (Markdown) generated successfully.")
