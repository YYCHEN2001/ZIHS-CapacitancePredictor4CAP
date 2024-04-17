from xgboost import XGBRegressor
from src.models import (train_evaluate, plot_actual_vs_predicted,
                        kfold_cv, load_data_dopants,
                        dataset_split_10class, model_results_to_md)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型，这里使用XGBoost回归器
xgb_regressor = XGBRegressor(random_state=21)

# 评估
results = train_evaluate(xgb_regressor, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, xgb_regressor.predict(X_train_scaled),
                         y_test, xgb_regressor.predict(X_test_scaled),
                         '../../../reports/figures/actual vs pred fig for data_dopants/xgb.png')

# K-fold 交叉验证
kfold_df = kfold_cv(xgb_regressor, X_train_scaled, y_train, n_splits=10, random_state=21)

# 显示交叉验证结果
print(kfold_df)

# 输出markdown报告
md_path = '../../../reports/results for data_dopants/xgb_results.md'
model_results_to_md(xgb_regressor, results, kfold_df, md_path)
