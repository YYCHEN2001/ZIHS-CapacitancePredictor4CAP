import pandas as pd
from xgboost import XGBRegressor
from src.models import (train_evaluate, plot_actual_vs_predicted,
                        kfold_cv, load_data_dopants,
                        dataset_split_10class, model_results_to_md)

# 读取数据
filepath = '../../../data/processed/data_8features.csv'
data = pd.read_csv(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型，这里使用XGBoost回归器
xgb_regressor = XGBRegressor(n_estimators=150,
                             learning_rate=0.2,
                             subsample=0.4,
                             gamma=0.8,
                             max_depth=9,
                             min_child_weight=2,
                             reg_alpha=0.14,
                             colsample_bytree=0.7,
                             colsample_bylevel=0.5,
                             colsample_bynode=1,
                             random_state=21)

# 评估
results = train_evaluate(xgb_regressor, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, xgb_regressor.predict(X_train_scaled),
                         y_test, xgb_regressor.predict(X_test_scaled),
                         '../../../reports/figures/actual vs pred fig for data_8features/xgb.png')

# K-fold 交叉验证
kfold_df = kfold_cv(xgb_regressor, X_train_scaled, y_train, n_splits=10, random_state=21)

# 显示交叉验证结果
print(kfold_df)

# 输出markdown报告
md_path = '../../../reports/results for data_8features/xgb_results.md'
model_results_to_md(xgb_regressor, results, kfold_df, md_path)
