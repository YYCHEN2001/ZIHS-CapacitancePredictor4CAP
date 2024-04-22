import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../../data/processed/data_8features.csv'
data = pd.read_csv(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

def objective(params):
    xgb = XGBRegressor(
        n_estimators=int(params['n_estimators']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        max_depth=int(params['max_depth']),
        gamma=params['gamma'],
        reg_alpha=params['reg_alpha'],
        min_child_weight=int(params['min_child_weight']),
        colsample_bytree=params['colsample_bytree'],
        colsample_bylevel=params['colsample_bylevel'],
        colsample_bynode=params['colsample_bynode'],
        random_state=21
    )
    metric = cross_val_score(xgb, X_train_scaled, y_train, cv=3, scoring='neg_root_mean_squared_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


space = {
    'n_estimators': hp.quniform('n_estimators', 10, 200, 10),
    'learning_rate': hp.quniform('learning_rate', 0.05, 0.3, 0.05),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'subsample': hp.quniform('subsample', 0.1, 1, 0.1),
    'gamma': hp.quniform('gamma', 0.1, 1.0, 0.1),
    'reg_alpha': hp.quniform('reg_alpha', 0.01, 1, 0.01),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1, 0.1),
    'colsample_bylevel': hp.quniform('colsample_bylevel', 0.1, 1, 0.1),
    'colsample_bynode': hp.quniform('colsample_bynode', 0.1, 1, 0.1)
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
