import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from src.models import plot_actual_vs_predicted, load_data_dopants, dataset_split_10class

# 读取数据
filepath = '../../../data/processed/data_8features.csv'
data = pd.read_csv(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化ANN模型
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # 输出层：一个神经元，无激活函数，用于回归任务
])

# 编译模型，指定优化器、损失函数和评价指标
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 初始化早停回调
early_stopper = EarlyStopping(
    monitor='val_loss',     # 监控验证集损失
    min_delta=0.01,        # 表示监控指标至少需要改善 0.001
    patience=50,            # 如果30个epoch内验证集损失没有改善，则提前停止训练
    verbose=1,              # 输出早停信息
    mode='min',             # 监控的指标是损失，应该减小
    restore_best_weights=True  # 训练结束后，模型权重回滚到最佳状态
)

# 训练模型
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.3,  # 使用20%的数据作为验证集
    epochs=500,  # 最大训练轮数
    callbacks=[early_stopper],  # 使用早停机制
    verbose=1  # 输出训练信息
)

# 预测训练集和测试集
y_train_pred = model.predict(X_train_scaled).flatten()
y_test_pred = model.predict(X_test_scaled).flatten()

# 计算训练集和测试集的R², MAE, MAPE, RMSE
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
train_rmse = root_mean_squared_error(y_train, y_train_pred)
test_rmse = root_mean_squared_error(y_test, y_test_pred)

# 创建一个DataFrame来保存所有指标
metrics_df = pd.DataFrame({
    "Metric": ["R²", "MAE", "MAPE", "RMSE"],
    "Train": [train_r2, train_mae, train_mape, train_rmse],
    "Test": [test_r2, test_mae, test_mape, test_rmse]
})

# 打印评估结果
print(metrics_df)

# 绘制训练集和测试集的实际值与预测值
plot_actual_vs_predicted(y_train, y_train_pred,
                         y_test, y_test_pred,
                         '../../../reports/figures/actual vs pred fig for data_8features/ann.png')

# 获取模型配置
model_config = model.get_config()

# 提取模型层和参数
layers = []
for layer in model_config['layers']:
    layer_name = layer['class_name']
    layer_params = layer['config']
    layers.append({
        'Layer': layer_name,
        'Output Shape': layer_params.get('units', 'N/A'),
        'Activation': layer_params.get('activation', 'N/A')
    })

# 创建模型参数DataFrame
params_df = pd.DataFrame(layers)

# 将DataFrame转换为Markdown字符串
metrics_md = metrics_df.to_markdown(index=False)
params_md = params_df.to_markdown(index=False)

# 将Markdown字符串写入文件
with open('../../../reports/results for data_8features/ann_results.md', 'w', encoding='utf-8') as f:
    f.write("# Model Parameters:\n")
    f.write(params_md + "\n\n")
    f.write("# Performance Metrics:\n")
    f.write(metrics_md)
