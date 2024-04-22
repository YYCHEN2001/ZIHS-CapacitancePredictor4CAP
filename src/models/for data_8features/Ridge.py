import pandas as pd
from sklearn.linear_model import Ridge
from src.models import (train_evaluate, plot_actual_vs_predicted,
                        kfold_cv, load_data_dopants,
                        dataset_split_10class, model_results_to_md)

# 读取数据
filepath = '../../../data/processed/data_8features.csv'
data = pd.read_csv(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型，这里使用Ridge回归
ridge = Ridge(alpha=0.8)

# 评估
results = train_evaluate(ridge, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, ridge.predict(X_train_scaled),
                         y_test, ridge.predict(X_test_scaled),
                         '../../../reports/figures/actual vs pred fig for data_8features/ridge.png')

kfold_df = kfold_cv(ridge, X_train_scaled, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(kfold_df)

# 输出markdown报告
md_path = '../../../reports/results for data_8features/ridge_results.md'
model_results_to_md(ridge, results, kfold_df, md_path)
