import pandas as pd


def generate_missing_values_report(data_path, report_path):
    # 加载数据
    data = pd.read_csv(data_path)

    # 检查各个特征的缺失值
    missing_values_report = data.isnull().sum().reset_index()
    missing_values_report.columns = ['Feature', 'Missing Values']
    missing_values_report['Missing Percentage'] = (missing_values_report['Missing Values'] / len(data)) * 100

    # 保存报告为Markdown格式
    with open(report_path, 'w') as f:
        f.write('# Missing Values Report\n\n')
        f.write('| Feature | Missing Values | Missing Percentage |\n')
        f.write('|---------|----------------|--------------------|\n')
        for _, row in missing_values_report.iterrows():
            f.write(f"| {row['Feature']} | {row['Missing Values']} | {row['Missing Percentage']:.2f}% |\n")


if __name__ == "__main__":
    data_path = '../../data/raw/cleaned_data.csv'  # 请将此路径替换为您的清洗后数据文件的实际路径
    report_path = '../../reports/data/missing_values_report.md'  # 请将此路径替换为您希望保存报告的实际路径
    generate_missing_values_report(data_path, report_path)
    print("Missing values report generated successfully.")
