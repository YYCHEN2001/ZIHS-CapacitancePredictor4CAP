import pandas as pd


def calculate_dopants(df):
    if df['B'] == 0 and df['P'] == 0 and df['S'] == 0:
        return 0
    else:
        return 1


if __name__ == "__main__":
    # 加载数据集
    data = pd.read_csv('../../data/raw/cleaned_data_after_mvp.csv')
    # 应用函数并创建新列
    data['Dopants'] = data.apply(calculate_dopants, axis=1)
    # 删除原始列
    data = data.drop(columns=['B', 'P', 'S'])
    # 保存数据集
    data.to_csv('../../data/processed/data_dopants.csv', index=False)
