import pandas as pd
import numpy as np


def detect_and_report_outliers(data_path, report_path):
    data = pd.read_csv(data_path)

    # 排除分类特征
    numeric_data = data.select_dtypes(include=[np.number])

    # 计算Z-score
    z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()
    outliers = (np.abs(z_scores) > 3)

    # 初始化报告内容
    report_content = "# Outliers Report Based on Z-score (Excluding Categorical Columns)\n\n"

    # 检测每个特征的异常值
    for column in outliers.columns:
        outlier_indices = outliers[outliers[column]].index.tolist()
        if outlier_indices:
            report_content += f"## {column}\nOutlier Rows Indices: {outlier_indices}\n\n"

    # 保存报告为Markdown格式
    with open(report_path, 'w') as f:
        f.write(report_content)


if __name__ == "__main__":
    data_path = '../../data/raw/cleaned_data_after_mvp.csv'  # 修改为您的文件路径
    report_path = '../../reports/data/outliers_report.md'  # 修改为您的报告保存路径
    detect_and_report_outliers(data_path, report_path)
    print("Outliers report generated successfully.")
