import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
svr = SVR()


# 定义目标函数
def objective(params):
    svr = SVR(
        C=params['C'],
        kernel=params['kernel'],
        degree=int(params['degree']),
        gamma=params['gamma'],
        coef0=params['coef0'],
        epsilon=params['epsilon']
    )
    metric = cross_val_score(svr, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error').mean()
    return {'loss': -metric, 'status': STATUS_OK}


# 定义超参数的搜索空间
space = {'gamma': 'scale',
         'C': hp.uniform('C', 0.1, 10),
         'degree': hp.choice('degree', [1, 2, 3, 4, 5, 6, 7, 8]),
         'coef0': hp.uniform('coef0', 0, 10),
         'epsilon': hp.uniform('epsilon', 0, 1),
         'kernel': 'poly'}

# 使用TPE算法进行优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

# 将best字典转换为DataFrame
best_df = pd.DataFrame([best])

# 使用round函数将每个超参数值四舍五入到两位小数
best_df = best_df.round(2)

print(best_df)
