import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
rfr = RandomForestRegressor(random_state=21)


# 定义目标函数
def objective(params):
    rfr = RandomForestRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_leaf=int(params['min_samples_leaf']),
        min_samples_split=int(params['min_samples_split']),
        random_state=21
    )
    metric = cross_val_score(rfr, X_train_scaled, y_train, cv=10, scoring='neg_root_mean_squared_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# Define the hyperparameter configuration space
space = {'n_estimators': hp.quniform('n_estimators', 10, 200, 10),
         'max_depth': hp.quniform('max_depth', 3, 15, 1),
         'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
         'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)}

# 使用TPE算法进行优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=300, trials=trials)

# 设置display.max_columns为None
pd.set_option('display.max_columns', None)

# 将best字典转换为DataFrame
best_df = pd.DataFrame([best])

# 使用round函数将每个超参数值四舍五入到两位小数
best_df = best_df.round(2)

print(best_df)
