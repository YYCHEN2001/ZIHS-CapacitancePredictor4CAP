import pandas as pd

def handle_missing_values(data_path, report_path):
    # 加载数据
    data = pd.read_csv(data_path)

    # 步骤1: 删除具有指定列缺失值的行，并记录被删除的行索引
    columns_no_missing = ['O', 'N', 'P', 'S', 'B', 'Specific surface area', 'Pore volume', 'Rmic/mes', 'ID/IG', 'Potential window', 'Current density']
    missing_rows = data[data[columns_no_missing].isnull().any(axis=1)]
    data = data.dropna(subset=columns_no_missing).copy()  # 显式创建副本

    # 步骤2: 使用'Active mass loading'的中位数填补其缺失值，并记录被填补的行索引
    aml_median = data['Active mass loading'].median()
    missing_aml_indices = data[data['Active mass loading'].isnull()].index
    data.fillna({'Active mass loading': aml_median}, inplace=True)

    # 准备报告内容
    missing_rows_report = missing_rows['Index'].to_markdown(index=False)
    filled_aml_report = pd.Series(missing_aml_indices).to_markdown(index=False)
    report_content = f"# Missing Value Processing Report\n\n## Rows Deleted\n{missing_rows_report}\n\n## Rows Filled ('Active mass loading')\n{filled_aml_report}"

    # 保存报告为Markdown格式
    with open(report_path, 'w') as f:
        f.write(report_content)

    # 返回清理后的数据
    return data

if __name__ == "__main__":
    data_path = '../../data/raw/cleaned_data.csv'
    report_path = '../../reports/data/missing_values_report.md.md'
    cleaned_data = handle_missing_values(data_path, report_path)
    print("Missing value processing completed. Report generated successfully.")
    # 可选: 保存清理后的数据
    cleaned_data_path = '../../data/raw/cleaned_data_after_mvp.csv'
    cleaned_data.to_csv(cleaned_data_path, index=False)
