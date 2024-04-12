import pandas as pd


def clean_data(file_path):
    # 加载数据
    data = pd.read_excel(file_path, sheet_name=0)
    # 删除第二行（单位行）和不需要的列
    data = data.drop(index=0).reset_index(drop=True)
    data = data.drop(columns=['Cathode', 'Ref'])

    # 确保特定列为数值类型
    numerical_columns = ['O', 'N', 'B', 'S', 'P', 'Specific surface area', 'Pore volume', 'Rmic/mes', 'ID/IG',
                         'Active mass loading', 'Potential window', 'Current density', 'Specific capacity']
    for col in numerical_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # 将'Specific capacity'列重命名为'target'
    data = data.rename(columns={'Specific capacity': 'target'})

    return data


def generate_feature_distribution_stats_md(data_path, report_path):
    # 加载数据
    data = pd.read_csv(data_path)

    # 为所有连续变量生成描述性统计
    continuous_stats = data.describe()

    # 为所有分类变量生成描述性统计
    categorical_columns = data.select_dtypes(include=['object']).columns
    categorical_stats = data[categorical_columns].describe()

    # 转换为Markdown格式
    stats_md = continuous_stats.to_markdown() + '\n\n' + categorical_stats.to_markdown()

    # 保存报告为Markdown格式
    with open(report_path, 'w') as f:
        f.write(stats_md)


if __name__ == "__main__":
    # 假定文件路径
    file_path = '../../data/raw/Carbon.xlsx'
    cleaned_data = clean_data(file_path)
    # 展示清洗后的数据前几行
    print(cleaned_data.head())
    # 保存清洗后的数据到新的Excel或CSV文件
    cleaned_data.to_csv('../../data/raw/cleaned_data.csv', index=False)
    # 生成特征分布统计报告
    data_path = '../../data/raw/cleaned_data.csv'
    report_path = '../../reports/data/feature_distribution_stats.md'
    generate_feature_distribution_stats_md(data_path, report_path)
    print("Feature distribution statistics report (Markdown) generated successfully.")
