import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 读取数据
data = pd.read_csv('../../data/processed/data_dopants.csv')
# 对分类变量进行独热编码
data_encoded = pd.get_dummies(data, columns=['Electrolyte', 'Current collector'])
# 删除索引列
data_encoded = data_encoded.drop('Index', axis=1)
# 将目标值分成10个等级
data_encoded['target_class'] = pd.qcut(data_encoded['target'], q=10, labels=False)
X = data_encoded.drop(['target', 'target_class'], axis=1)
y = data_encoded['target']
stratify_column = data_encoded['target_class']

# 拆分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, random_state=21,
                                                    stratify=stratify_column)

# 初始化标准化器
scaler = StandardScaler()

# 使用训练集拟合标准化器
scaler.fit(X_train)

# 使用拟合过的标准化器来转换训练集和测试集
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化ANN模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  # 输出层：一个神经元，无激活函数，用于回归任务
])

# 编译模型，指定优化器、损失函数和评价指标
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2, verbose=1)

# 评估模型
train_mse, train_mae = model.evaluate(X_train_scaled, y_train, verbose=0)
test_mse, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
results = {'train_mse': train_mse, 'train_mae': train_mae, 'test_mse': test_mse, 'test_mae': test_mae}

# 打印评估结果
print(results)
