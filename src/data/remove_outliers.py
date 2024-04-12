import pandas as pd
import numpy as np
from feature_distribution_stats_before_mvp import generate_feature_distribution_stats_md


def remove_outliers(data_path, output_path):
    # 加载数据集
    data = pd.read_csv(data_path)

    # 识别数值型特征列
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    # 计算数值型特征的Z-score
    z_scores = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()

    # 根据Z-score > 3的规则找到异常值索引
    outliers = (np.abs(z_scores) > 3).any(axis=1)

    # 在源数据集的完整副本上删除包含异常值的行
    data_cleaned = data[~outliers].copy()

    # 保存处理后的数据集为新的CSV文件
    data_cleaned.to_csv(output_path, index=False)


if __name__ == "__main__":
    data_path = '../../data/raw/cleaned_data_after_mvp.csv'  # 替换为您的数据集路径
    output_path = '../../data/processed/cleaned_data_no_outliers.csv'  # 替换为您希望保存新数据集的路径
    remove_outliers(data_path, output_path)
    print("已成功移除异常值，并保存了新的数据集。")
    data_path_1 = '../../data/processed/cleaned_data_no_outliers.csv'
    report_path_1 = '../../reports/data/feature_distribution_stats_after_outliers_removal.md'
    generate_feature_distribution_stats_md(data_path_1, report_path_1)
    print("特征分布统计报告（Markdown）已成功生成。")