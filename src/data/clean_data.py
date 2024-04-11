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


# 假定文件路径
file_path = '../../data/raw/Carbon.xlsx'
cleaned_data = clean_data(file_path)

# 展示清洗后的数据的前几行
print(cleaned_data.head())

# 保存清洗后的数据到新的Excel或CSV文件
cleaned_data.to_csv('../../data/raw/cleaned_data.csv', index=False)
