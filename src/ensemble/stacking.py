# src/ensemble/stacking.py
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import RegressorMixin, clone
from sklearn.multioutput import MultiOutputRegressor 
import numpy as np
import pandas as pd

from src import config # 导入配置文件
from src.models.model_provider import MODEL_GETTERS # 导入模型获取器

def train_stacking_model_custom(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    base_learner_configs: list[tuple[str, dict | None]], 
    meta_learner_config: tuple[str, dict | None], 
    cv_folds: int = None,
    use_sklearn_stacking: bool = True 
) -> RegressorMixin:
    """
    训练Stacking集成模型。
    对于多目标输出，会自动使用 MultiOutputRegressor 包装 StackingRegressor。

    参数:
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.DataFrame): 训练目标 (多输出)。
        base_learner_configs (list[tuple[str, dict | None]]): 
            一级学习器的配置列表。每个元组包含 (模型名称, 参数字典或None)。
        meta_learner_config (tuple[str, dict | None]): 
            二级学习器（元学习器）的配置。元组包含 (模型名称, 参数字典或None)。
        cv_folds (int, optional): 交叉验证的折数。如果为None，则使用config中的默认值。
        use_sklearn_stacking (bool): （在单目标时）是否使用 scikit-learn 内置的 StackingRegressor。
                                     对于多目标，总是基于 StackingRegressor 构建。

    返回:
        RegressorMixin: 训练好的Stacking模型。
    """
    if cv_folds is None:
        cv_folds = config.DEFAULT_CV_FOLDS
    
    print(f"开始Stacking模型训练。")
    print(f"一级学习器数量: {len(base_learner_configs)}, 元学习器: {meta_learner_config[0]}, CV折数: {cv_folds}")

    # 1. 初始化一级学习器
    base_learners_estimators = []
    for name, params in base_learner_configs:
        if name not in MODEL_GETTERS:
            raise ValueError(f"未知的基学习器名称: {name}。请检查 model_provider.py")
        estimator = MODEL_GETTERS[name](params)
        base_learners_estimators.append((name, estimator))

    # 2. 初始化元学习器
    meta_name, meta_params = meta_learner_config
    if meta_name not in MODEL_GETTERS:
        raise ValueError(f"未知的元学习器名称: {meta_name}。请检查 model_provider.py")
    meta_learner_estimator = MODEL_GETTERS[meta_name](meta_params)

    # 3. 数据准备
    if isinstance(y_train, pd.DataFrame):
        y_train_np = y_train.to_numpy()
    elif isinstance(y_train, pd.Series): 
        y_train_np = y_train.to_numpy().reshape(-1, 1)
    else:
        y_train_np = y_train 

    if y_train_np.ndim == 1: 
        y_train_np = y_train_np.reshape(-1, 1)
    
    print(f"训练数据形状: X_train: {X_train.shape}, y_train_np: {y_train_np.shape}")

    # 4. Stacking 模型训练
    if use_sklearn_stacking: 
        if y_train_np.shape[1] > 1: 
            print("检测到多目标输出。将使用 MultiOutputRegressor 包装 StackingRegressor。")
            
            inner_stacker = StackingRegressor(
                estimators=base_learners_estimators,
                final_estimator=meta_learner_estimator,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE),
                n_jobs=1,  
                passthrough=False
            )
            
            # 如果遇到 ResourceTracker.__del__ 相关的 AttributeError,
            # 可以尝试将下面的 n_jobs=-1 改为 n_jobs=1 进行诊断。
            # 这会禁用 MultiOutputRegressor 的并行处理，可能解决清理阶段的错误，但会降低训练速度。
            stacking_model = MultiOutputRegressor(inner_stacker, n_jobs=1) # 或者 n_jobs=1 用于诊断
            
            print("正在使用 MultiOutputRegressor(StackingRegressor) 拟合模型...")
            stacking_model.fit(X_train, y_train_np) 

        else: 
            print("检测到单目标输出。直接使用 StackingRegressor。")
            stacking_model = StackingRegressor(
                estimators=base_learners_estimators,
                final_estimator=meta_learner_estimator,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE),
                n_jobs=-1, 
                passthrough=False
            )
            print("正在使用 StackingRegressor 拟合模型...")
            stacking_model.fit(X_train, y_train_np.ravel())
    
    else: 
        print("手动执行Stacking流程 (多目标处理可能不完整)...")
        meta_features_list = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

        for i, (name, model) in enumerate(base_learners_estimators):
            print(f"  为基学习器 '{name}' 生成元特征...")
            oof_preds = cross_val_predict(model, X_train, y_train_np, cv=kf, n_jobs=-1)
            if oof_preds.ndim == 1: 
                oof_preds = oof_preds.reshape(-1, 1)
            meta_features_list.append(oof_preds)
            print(f"    '{name}' 的元特征形状: {oof_preds.shape}")

        meta_X = np.concatenate(meta_features_list, axis=1)
        print(f"  合并后的元特征形状 (meta_X): {meta_X.shape}")

        print(f"  训练元学习器 '{meta_name}'...")
        meta_learner_estimator.fit(meta_X, y_train_np)
        
        print("  在全部训练数据上重新训练基学习器...")
        final_base_learners = []
        for name, model_template in base_learners_estimators:
            fitted_model = clone(model_template).fit(X_train, y_train_np)
            final_base_learners.append(fitted_model)
        
        class ManualStacker:
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model
            
            def predict(self, X_new):
                base_predictions = []
                for model in self.base_models:
                    pred = model.predict(X_new)
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                    base_predictions.append(pred)
                meta_X_new = np.concatenate(base_predictions, axis=1)
                return self.meta_model.predict(meta_X_new)
            
            def get_params(self, deep=True): 
                return {"base_models": self.base_models, "meta_model": self.meta_model}

        stacking_model = ManualStacker(final_base_learners, meta_learner_estimator)
        print("手动Stacking模型组装完成。")

    print("Stacking模型训练完成。")
    return stacking_model

if __name__ == '__main__':
    from src.data_processing.loader import get_processed_data 
    
    print("--- 测试Stacking集成模块 ---")
    try:
        X_train_data, y_train_data, X_test_data, y_test_data, _ = get_processed_data(scale_features=False)

        if X_train_data.empty or y_train_data.empty:
            raise ValueError("测试Stacking时，数据加载失败。")

        base_configs = [
            ('RF', {'n_estimators': 20, 'max_depth': 5, 'random_state': config.RANDOM_STATE}),
            ('XGBR', {'n_estimators': 20, 'max_depth': 3, 'random_state': config.RANDOM_STATE}),
        ]
        meta_config = ('LR', None) 

        print("\n--- 测试使用 scikit-learn StackingRegressor (已适配多目标) ---")
        trained_stacking_model_sklearn = train_stacking_model_custom(
            X_train_data, y_train_data, base_configs, meta_config, cv_folds=3, use_sklearn_stacking=True
        )
        print("Scikit-learn Stacking模型详情:")
        print(trained_stacking_model_sklearn)

        if not X_test_data.empty:
            sample_predictions_sklearn = trained_stacking_model_sklearn.predict(X_test_data.head())
            print("\n使用Scikit-learn Stacking模型在测试集头部的样本预测:")
            print(sample_predictions_sklearn)
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"数值错误: {e}")
    except Exception as e:
        print(f"测试Stacking模块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
