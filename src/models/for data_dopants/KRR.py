from sklearn.kernel_ridge import KernelRidge
from src.models import (train_evaluate, plot_actual_vs_predicted,
                        kfold_cv, load_data_dopants,
                        dataset_split_10class, model_results_to_md)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
krr = KernelRidge(alpha=0.8,
                  gamma=0.1,
                  kernel='polynomial',
                  degree=2,
                  coef0=7.7)

# 评估
results = train_evaluate(krr, X_train_scaled, y_train, X_test_scaled, y_test)
print(results)

# 绘制实际值与预测值
plot_actual_vs_predicted(y_train, krr.predict(X_train_scaled),
                         y_test, krr.predict(X_test_scaled),
                         '../../../reports/figures/actual vs pred fig for data_dopants/krr.png')

kfold_df = kfold_cv(krr, X_train_scaled, y_train, n_splits=10, random_state=21)

# Display the metrics for each fold and the averages
print(kfold_df)

# 输出markdown报告
md_path = '../../../reports/results for data_dopants/krr_results.md'
model_results_to_md(krr, results, kfold_df, md_path)
