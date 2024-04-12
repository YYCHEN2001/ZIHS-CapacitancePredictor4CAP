from feature_distribution_stats_before_mvp import generate_feature_distribution_stats_md

# 加载数据
data_path = '../../data/raw/cleaned_data_after_mvp.csv'
report_path = '../../reports/data/feature_distribution_stats_after_mvp.md'
generate_feature_distribution_stats_md(data_path, report_path)
print("Feature distribution statistics report (Markdown) generated successfully.")
