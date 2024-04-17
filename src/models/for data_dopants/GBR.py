from sklearn.ensemble import GradientBoostingRegressor

from src.models import (train_evaluate, plot_actual_vs_predicted,
                        kfold_cv, load_data_dopants,
                        dataset_split_10class, model_results_to_md)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
gbr = GradientBoostingRegressor(n_estimators=200,
                                alpha=0.07,
                                learning_rate=0.14,
                                max_depth=9,
                                max_features=0.2,
                                min_samples_leaf=3,
                                min_samples_split=7,
                                subsample=0.8,
                                random_state=21)

# 评估
results = train_evaluate(gbr, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, gbr.predict(X_train_scaled),
                         y_test, gbr.predict(X_test_scaled),
                         '../../../reports/figures/actual vs pred fig for data_dopants/gbr.png')

kfold_df = kfold_cv(gbr, X_train_scaled, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(kfold_df)

# 输出markdown报告
md_path = '../../../reports/results for data_dopants/rfr_results.md'
model_results_to_md(gbr, results, kfold_df, md_path)
