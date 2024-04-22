from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from src.models import dataset_split_10class
import pandas as pd

# 读取数据
filepath = '../../../data/processed/data_8features.csv'
data = pd.read_csv(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
rfr = RandomForestRegressor(n_estimators=200,
                            random_state=21)

# 定义参数网格
param_grid = {
    # 'n_estimators': [100, 200, 300, 400],
    'max_depth': range(12, 20),
    'min_samples_leaf': [1, 2, 3, 4],
    'min_samples_split': [2, 3, 4, 5]
}

# 初始化网格搜索
grid_search = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2,
                           n_jobs=-1)

# 执行网格搜索
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数和对应的评分
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)