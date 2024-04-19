import shap
from xgboost import XGBRegressor

from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型，这里使用XGBoost回归器
xgb = XGBRegressor(n_estimators=190,
                   learning_rate=0.15,
                   subsample=0.5,
                   gamma=0.1,
                   max_depth=8,
                   min_child_weight=2,
                   reg_alpha=0.34,
                   colsample_bytree=1.0,
                   colsample_bylevel=0.3,
                   colsample_bynode=0.7,
                   random_state=21)

xgb.fit(X_train_scaled, y_train)
y_pred_train = xgb.predict(X_train_scaled)
y_pred_test = xgb.predict(X_test_scaled)

# Calculate SHAP values
explainer = shap.Explainer(xgb)
shap_values = explainer(X_train_scaled)

# Plot summary of SHAP values
shap.summary_plot(shap_values, X_train_scaled, plot_type="bar", plot_size=(12, 10))
shap.summary_plot(shap_values, X_train_scaled, plot_size=(25.6, 14.4))
