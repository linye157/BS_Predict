# src/ensemble/stacking.py
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.base import RegressorMixin, clone
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
    可以选择使用scikit-learn的StackingRegressor或一个简化的手动实现流程。

    参数:
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.DataFrame): 训练目标 (多输出)。
        base_learner_configs (list[tuple[str, dict | None]]): 
            一级学习器的配置列表。每个元组包含 (模型名称, 参数字典或None)。
            例如: [('RF', {'n_estimators': 50}), ('XGBR', None)]
        meta_learner_config (tuple[str, dict | None]): 
            二级学习器（元学习器）的配置。元组包含 (模型名称, 参数字典或None)。
            例如: ('LR', None)
        cv_folds (int, optional): 交叉验证的折数。如果为None，则使用config中的默认值。
        use_sklearn_stacking (bool): 是否使用 scikit-learn 内置的 StackingRegressor。
                                     True (默认) 使用 StackingRegressor。
                                     False 使用一个简化的手动 k-fold 预测流程生成元特征。

    返回:
        RegressorMixin: 训练好的Stacking模型。
    """
    if cv_folds is None:
        cv_folds = config.DEFAULT_CV_FOLDS
    
    print(f"开始Stacking模型训练。使用 {'scikit-learn StackingRegressor' if use_sklearn_stacking else '手动流程'}。")
    print(f"一级学习器数量: {len(base_learner_configs)}, 元学习器: {meta_learner_config[0]}, CV折数: {cv_folds}")

    # 1. 初始化一级学习器
    base_learners_estimators = []
    for name, params in base_learner_configs:
        if name not in MODEL_GETTERS:
            raise ValueError(f"未知的基学习器名称: {name}。请检查 model_provider.py")
        estimator = MODEL_GETTERS[name](params)
        base_learners_estimators.append((name, estimator)) # StackingRegressor 需要 (name, estimator) 格式

    # 2. 初始化元学习器
    meta_name, meta_params = meta_learner_config
    if meta_name not in MODEL_GETTERS:
        raise ValueError(f"未知的元学习器名称: {meta_name}。请检查 model_provider.py")
    meta_learner_estimator = MODEL_GETTERS[meta_name](meta_params)

    # 确保 y_train 是 NumPy 数组，以便与 scikit-learn 更好地兼容
    # StackingRegressor 内部会处理多输出的 y
    if isinstance(y_train, pd.DataFrame):
        y_train_np = y_train.to_numpy()
    elif isinstance(y_train, pd.Series): # 单目标情况
        y_train_np = y_train.to_numpy().reshape(-1, 1)
    else:
        y_train_np = y_train # 假设已经是numpy数组

    if y_train_np.ndim == 1: # 确保y是二维的 (n_samples, n_targets)
        y_train_np = y_train_np.reshape(-1, 1)
    
    print(f"训练数据形状: X_train: {X_train.shape}, y_train_np: {y_train_np.shape}")


    if use_sklearn_stacking:
        # 使用 scikit-learn 的 StackingRegressor
        # 它内部处理交叉验证以生成元特征
        # passthrough=False: 只使用基模型的预测作为元学习器的输入
        # passthrough=True: 原始特征也会被加入到元学习器的输入中
        stacking_model = StackingRegressor(
            estimators=base_learners_estimators,
            final_estimator=meta_learner_estimator,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE),
            n_jobs=-1, # 使用所有可用核心
            passthrough=False 
        )
        print("正在使用 StackingRegressor 拟合模型...")
        stacking_model.fit(X_train, y_train_np) # StackingRegressor 可以直接处理 DataFrame X 和 NumPy y

    else:
        # 手动实现Stacking流程 (简化版，主要用于演示概念)
        print("手动执行Stacking流程...")
        meta_features_list = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

        for i, (name, model) in enumerate(base_learners_estimators):
            print(f"  为基学习器 '{name}' 生成元特征...")
            # 使用 cross_val_predict 获取每个基学习器在交叉验证中的预测
            # 这将作为元学习器的输入特征
            # y_train_np[:, 0] 是一个示例，如果y是多输出，需要为每个输出或整体进行预测
            # cross_val_predict 对多输出y的处理取决于estimator是否原生支持
            # 如果estimator不支持，可能需要对每个目标单独调用或使用MultiOutputRegressor包装
            
            # 假设基学习器都能处理多输出的y_train_np
            # cross_val_predict 会为每个目标生成预测（如果模型支持）
            # 结果 oof_preds 的形状将是 (n_samples, n_targets)
            oof_preds = cross_val_predict(model, X_train, y_train_np, cv=kf, n_jobs=-1)
            if oof_preds.ndim == 1: # 如果是单目标或模型只返回一维
                oof_preds = oof_preds.reshape(-1, 1)
            meta_features_list.append(oof_preds)
            print(f"    '{name}' 的元特征形状: {oof_preds.shape}")

        # 合并所有基学习器的预测作为元学习器的输入特征
        meta_X = np.concatenate(meta_features_list, axis=1)
        print(f"  合并后的元特征形状 (meta_X): {meta_X.shape}")

        # 训练元学习器
        print(f"  训练元学习器 '{meta_name}'...")
        meta_learner_estimator.fit(meta_X, y_train_np)
        
        # 为了能像scikit-learn的stacker一样使用，我们需要一个包装器
        # 这里简化，直接返回训练好的元学习器，并假设预测时也会先生成元特征
        # 一个更完整的实现会创建一个自定义的Stacking类
        # 这里我们直接将训练好的基学习器和元学习器组合起来用于预测
        # 首先，在全部训练数据上重新训练基学习器
        print("  在全部训练数据上重新训练基学习器...")
        final_base_learners = []
        for name, model_template in base_learners_estimators:
            fitted_model = clone(model_template).fit(X_train, y_train_np)
            final_base_learners.append(fitted_model)
        
        # 定义一个简单的类来模拟 StackingRegressor 的 predict 方法
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
            
            def get_params(self, deep=True): # 使得可以被GridSearchCV等工具使用
                return {"base_models": self.base_models, "meta_model": self.meta_model}


        stacking_model = ManualStacker(final_base_learners, meta_learner_estimator)
        print("手动Stacking模型组装完成。")

    print("Stacking模型训练完成。")
    return stacking_model

if __name__ == '__main__':
    from src.data_processing.loader import get_processed_data
    
    print("--- 测试Stacking集成模块 ---")
    try:
        # 1. 加载数据
        # 注意：这里不进行标准化，因为某些模型（如树模型）对标准化不敏感，
        # 而Stacking中混合不同模型时，标准化通常在每个模型内部或之前统一处理。
        # 如果需要，可以在get_processed_data中设置scale_features=True
        X_train_data, y_train_data, X_test_data, y_test_data, _ = get_processed_data(scale_features=False)

        if X_train_data.empty or y_train_data.empty:
            raise ValueError("测试Stacking时，数据加载失败。")

        # 2. 定义一级学习器配置
        # 使用较少的估计器和深度以加快测试速度
        base_configs = [
            ('RF', {'n_estimators': 20, 'max_depth': 5, 'random_state': config.RANDOM_STATE}),
            ('XGBR', {'n_estimators': 20, 'max_depth': 3, 'random_state': config.RANDOM_STATE}),
            # ('SVR', {'C': 0.5}) # SVR可能较慢，测试时可以注释掉
        ]

        # 3. 定义二级学习器（元学习器）配置
        meta_config = ('LR', None) # 使用线性回归作为元学习器

        # 4. 训练Stacking模型 (使用scikit-learn的StackingRegressor)
        print("\n--- 测试使用 scikit-learn StackingRegressor ---")
        trained_stacking_model_sklearn = train_stacking_model_custom(
            X_train_data, y_train_data, base_configs, meta_config, cv_folds=3, use_sklearn_stacking=True
        )
        print("Scikit-learn Stacking模型详情:")
        print(trained_stacking_model_sklearn)

        # 5. 进行简单预测 (可选)
        if not X_test_data.empty:
            sample_predictions_sklearn = trained_stacking_model_sklearn.predict(X_test_data.head())
            print("\n使用Scikit-learn Stacking模型在测试集头部的样本预测:")
            print(sample_predictions_sklearn)
        
        # 6. 训练Stacking模型 (使用手动流程，可选，用于对比或理解)
        # print("\n--- 测试使用手动 Stacking 流程 ---")
        # trained_stacking_model_manual = train_stacking_model_custom(
        #     X_train_data, y_train_data, base_configs, meta_config, cv_folds=3, use_sklearn_stacking=False
        # )
        # print("手动 Stacking 模型详情:")
        # print(trained_stacking_model_manual) # 如果是自定义类，可能没有详细的__repr__

        # if not X_test_data.empty and hasattr(trained_stacking_model_manual, 'predict'):
        #     sample_predictions_manual = trained_stacking_model_manual.predict(X_test_data.head())
        #     print("\n使用手动 Stacking 模型在测试集头部的样本预测:")
        #     print(sample_predictions_manual)


    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"数值错误: {e}")
    except Exception as e:
        print(f"测试Stacking模块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
