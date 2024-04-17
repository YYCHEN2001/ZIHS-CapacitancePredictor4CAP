import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data_dopants(filepath):
    # 读取数据
    data = pd.read_csv(filepath)
    # 对分类变量进行独热编码
    data_encoded = pd.get_dummies(data, columns=['Electrolyte', 'Current collector'])
    # 删除索引列
    data_encoded = data_encoded.drop('Index', axis=1)

    return data_encoded


def dataset_split_10class(data_encoded):
    # 将目标值分成10个等级
    data_encoded['target_class'] = pd.qcut(data_encoded['target'], q=10, labels=False)
    x = data_encoded.drop(['target', 'target_class'], axis=1)
    y = data_encoded['target']
    stratify_column = data_encoded['target_class']

    # 拆分训练和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3, random_state=21,
                                                        stratify=stratify_column)

    # 初始化标准化器
    scaler = StandardScaler()

    # 使用训练集拟合标准化器
    scaler.fit(x_train)

    # 使用拟合过的标准化器来转换训练集和测试集
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 将Series转换为DataFrame，这是为了kfold_cv函数的输入
    x_train_scaled = pd.DataFrame(x_train_scaled, columns=x.columns)
    x_test_scaled = pd.DataFrame(x_test_scaled, columns=x.columns)

    return x_train_scaled, x_test_scaled, y_train, y_test
