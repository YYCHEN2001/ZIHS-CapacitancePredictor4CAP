import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from src.models import (load_data_dopants, dataset_split_10class)

# 读取数据
filepath = '../../../data/processed/data_dopants.csv'
data = load_data_dopants(filepath)

# 按10个等级分割数据集，同时标准化数据
X_train_scaled, X_test_scaled, y_train, y_test = dataset_split_10class(data)

# 初始化模型
krr = KernelRidge()


# 定义目标函数
def objective(params):
    model = KernelRidge(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='neg_mean_absolute_error').mean()
    return {'loss': -score, 'status': STATUS_OK}


# 定义超参数的搜索空间
space = {
    'alpha': hp.loguniform('alpha', -5, 2),
    'gamma': hp.loguniform('gamma', -5, 2),
    'degree': hp.choice('degree', [2, 3, 4, 5]),
    'coef0': hp.uniform('coef0', 0, 10),
    'kernel': 'polynomial'
}

# 使用TPE算法进行优化
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=300, trials=trials)

# 使用np.exp函数将每个超参数值取指数
best_exp = {k: np.exp(v) if k in ['alpha', 'gamma'] else v for k, v in best.items()}

# 将best字典转换为DataFrame
best_df = pd.DataFrame([best_exp])

# 使用round函数将每个超参数值四舍五入到两位小数
best_df = best_df.round(2)

print(best_df)
