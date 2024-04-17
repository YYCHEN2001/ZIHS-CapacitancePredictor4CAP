import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
gbr = GradientBoostingRegressor(random_state=21)


# 定义目标函数
def objective(params):
    gbr = GradientBoostingRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        alpha=params['alpha'],
        min_samples_split=int(params['min_samples_split']),
        min_samples_leaf=int(params['min_samples_leaf']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_features=params['max_features'],
        random_state=21
    )
    metric = cross_val_score(gbr, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# 定义超参数的搜索空间
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'alpha': hp.uniform('alpha', 0.00001, 0.1),
    'max_depth': hp.quniform('max_depth', 1, 12, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.quniform('subsample', 0.1, 1.0, 0.1),
    'max_features': hp.uniform('max_features', 0.01, 1)
}

# 使用TPE算法进行优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=300, trials=trials)

# 设置display.max_columns为None
pd.set_option('display.max_columns', None)

# 将best字典转换为DataFrame
best_df = pd.DataFrame([best])

# 使用round函数将每个超参数值四舍五入到两位小数
best_df = best_df.round(3)

print(best_df)
