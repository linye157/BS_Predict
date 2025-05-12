# src/models/model_provider.py
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor # 用于包装不支持多输出的SVR
import xgboost as xgb

from src import config # 导入配置文件

# 为了代码整洁和类型提示
from sklearn.base import RegressorMixin 

def get_lr_model(params: dict = None) -> RegressorMixin:
    """
    获取线性回归 (Linear Regression) 模型实例。
    LinearRegression 本身支持多目标输出。

    参数:
        params (dict, optional): 模型的参数。如果为None，则使用默认参数。

    返回:
        RegressorMixin: scikit-learn 线性回归模型实例。
    """
    model_params = params if params is not None else {}
    print("创建线性回归模型...")
    return LinearRegression(**model_params)

def get_rf_model(params: dict = None) -> RegressorMixin:
    """
    获取随机森林回归 (Random Forest Regressor) 模型实例。
    RandomForestRegressor 本身支持多目标输出。

    参数:
        params (dict, optional): 模型的参数。如果为None，则使用config中定义的默认参数。

    返回:
        RegressorMixin: scikit-learn 随机森林回归模型实例。
    """
    model_params = config.RF_PARAMS.copy() # 获取默认参数副本
    if params:
        model_params.update(params) # 如果提供了参数，则更新默认参数
    print(f"创建随机森林回归模型，参数: {model_params}")
    return RandomForestRegressor(**model_params)

def get_gbr_model(params: dict = None) -> RegressorMixin:
    """
    获取梯度提升回归 (Gradient Boosting Regressor) 模型实例。
    GradientBoostingRegressor 本身支持多目标输出。

    参数:
        params (dict, optional): 模型的参数。如果为None，则使用默认参数。

    返回:
        RegressorMixin: scikit-learn 梯度提升回归模型实例。
    """
    # GBR没有在config中预设，这里给一些常用默认值
    default_gbr_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': config.RANDOM_STATE}
    if params:
        default_gbr_params.update(params)
    print(f"创建梯度提升回归模型，参数: {default_gbr_params}")
    return GradientBoostingRegressor(**default_gbr_params)

def get_xgbr_model(params: dict = None) -> RegressorMixin:
    """
    获取 XGBoost 回归 (XGBRegressor) 模型实例。
    XGBRegressor 本身支持多目标输出。

    参数:
        params (dict, optional): 模型的参数。如果为None，则使用config中定义的默认参数。

    返回:
        RegressorMixin: XGBoost 回归模型实例。
    """
    model_params = config.XGBR_PARAMS.copy()
    if params:
        model_params.update(params)
    print(f"创建XGBoost回归模型，参数: {model_params}")
    return xgb.XGBRegressor(**model_params)

def get_svr_model(params: dict = None) -> RegressorMixin:
    """
    获取支持向量回归 (Support Vector Regressor) 模型实例。
    SVR 本身不支持多目标输出，因此使用 MultiOutputRegressor 进行包装。

    参数:
        params (dict, optional): SVR核模型的参数。如果为None，则使用config中定义的默认参数。

    返回:
        RegressorMixin: 经过 MultiOutputRegressor 包装的SVR模型实例。
    """
    svr_base_params = config.SVR_PARAMS.copy()
    if params:
        svr_base_params.update(params)
    
    print(f"创建SVR模型 (使用MultiOutputRegressor包装)，基础SVR参数: {svr_base_params}")
    # SVR需要MultiOutputRegressor来处理多目标输出
    # MultiOutputRegressor会为每个目标列独立训练一个SVR模型
    return MultiOutputRegressor(SVR(**svr_base_params), n_jobs=config.RF_PARAMS.get('n_jobs', -1)) # 复用n_jobs设置

def get_ann_model(params: dict = None) -> RegressorMixin:
    """
    获取人工神经网络回归 (Artificial Neural Network - MLPRegressor) 模型实例。
    MLPRegressor 本身支持多目标输出。

    参数:
        params (dict, optional): 模型的参数。如果为None，则使用config中定义的默认参数。

    返回:
        RegressorMixin: scikit-learn MLPRegressor模型实例。
    """
    model_params = config.ANN_PARAMS.copy()
    if params:
        model_params.update(params)
    print(f"创建人工神经网络(MLPRegressor)模型，参数: {model_params}")
    return MLPRegressor(**model_params)

# 模型名称到获取函数的映射，方便在其他模块中通过名称调用
MODEL_GETTERS = {
    "LR": get_lr_model,
    "RF": get_rf_model,
    "GBR": get_gbr_model,
    "XGBR": get_xgbr_model,
    "SVR": get_svr_model,
    "ANN": get_ann_model,
}

if __name__ == '__main__':
    print("--- 测试模型提供模块 ---")
    
    # 测试获取所有定义的模型
    for model_name, getter_func in MODEL_GETTERS.items():
        print(f"\n尝试获取模型: {model_name}")
        try:
            model = getter_func()
            print(f"成功获取 {model_name} 模型: {model}")
            # 可以简单打印模型参数
            print(f"模型参数: {model.get_params()}")
        except Exception as e:
            print(f"获取 {model_name} 模型失败: {e}")

    # 测试带参数的模型获取
    print("\n--- 测试带自定义参数的模型获取 ---")
    custom_rf_params = {'n_estimators': 150, 'max_depth': 5}
    rf_custom = get_rf_model(params=custom_rf_params)
    print(f"自定义参数的RF模型: {rf_custom.get_params()}")

    custom_svr_params = {'C': 2.0, 'kernel': 'linear'}
    svr_custom = get_svr_model(params=custom_svr_params)
    # 对于MultiOutputRegressor，需要访问其estimator的参数
    if hasattr(svr_custom, 'estimator'):
        print(f"自定义参数的SVR (基础模型) 参数: {svr_custom.estimator.get_params()}")
    else:
        print(f"自定义参数的SVR模型: {svr_custom.get_params()}")

