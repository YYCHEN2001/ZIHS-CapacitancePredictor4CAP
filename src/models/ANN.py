import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from src.models import (load_data_dopants, dataset_split_10class,
                        plot_actual_vs_predicted, model_results_to_md)

# 读取数据
filepath = '../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

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

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, model.predict(X_train_scaled),
                         y_test, model.predict(X_test_scaled),
                         '../../reports/figures/model evaluation/ann.png')

# 不使用kfold_cv，因为ANN的训练方式与传统ML方法不同
# 如果需要k-fold验证，你需要重新定义交叉验证的逻辑

# 输出markdown报告
md_path = '../../reports/models/ann_results.md'
model_results_to_md(model, results, None, md_path)
