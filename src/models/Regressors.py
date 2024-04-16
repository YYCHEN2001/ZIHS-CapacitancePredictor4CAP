from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class Regressors:
    def __init__(self, krr_params=None, rfr_params=None, gbr_params=None):
        """
        初始化模型与参数
        :param krr_params: Kernel Ridge Regression 参数
        :param rfr_params: Random Forest Regression 参数
        :param gbr_params: Gradient Boosting Regression 参数
        """
        self.krr = KernelRidge(**(krr_params if krr_params is not None else {}))
        self.rfr = RandomForestRegressor(**(rfr_params if rfr_params is not None else {}))
        self.gbr = GradientBoostingRegressor(**(gbr_params if gbr_params is not None else {}))

