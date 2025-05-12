# src/automl/optimization.py
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score # 示例回归指标
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Callable

from src import config # 导入配置文件
from src.models import model_provider # 用于获取模型实例
from sklearn.base import RegressorMixin

def select_best_model_with_hyperparam_tuning(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    models_param_grids: Dict[str, Tuple[Callable[..., RegressorMixin], Dict[str, Any]]],
    cv_folds: int = None, 
    scoring: str | Callable | List[str] | Dict[str, Callable] = 'r2', # 默认使用 R^2 评分
    n_iter_random: int = 10, # RandomizedSearchCV的迭代次数
    use_random_search: bool = True, # True 使用 RandomizedSearchCV, False 使用 GridSearchCV
    refit_metric: str = None # 对于多指标评分，指定用于refit最佳模型的指标
) -> Tuple[str | None, RegressorMixin | None, Dict[str, Any] | None, float | Dict[str, float] | None]:
    """
    自动筛选模型并进行超参数优化。

    参数:
        X_train (pd.DataFrame): 训练特征。
        y_train (pd.DataFrame): 训练目标 (可以是多输出 DataFrame)。
        models_param_grids (Dict): 字典，键为模型名称，值为元组。
                                   元组包含 (模型获取函数, 参数网格字典)。
                                   例如: {'RF': (model_provider.get_rf_model, {'n_estimators': [50, 100]})}
        cv_folds (int, optional): 交叉验证折数。如果为None，则使用config中的默认值。
        scoring (str | Callable | List | Dict, optional): 用于评估的评分方法。
                 对于多输出，可以传入一个包含每个目标评分的列表，或者一个平均策略。
                 如果为None，则使用模型默认的score方法。默认为 'r2'。
        n_iter_random (int): 如果使用RandomizedSearchCV，这是参数设置的迭代次数。
        use_random_search (bool): 是否使用RandomizedSearchCV。True则使用，False则使用GridSearchCV。
        refit_metric (str, optional): 如果scoring是字典（多指标评估），此参数指定哪个指标用于 `best_estimator_` 的 refit。
                                      如果scoring是单一指标字符串或列表，则不需要此参数。

    返回:
        一个元组 (best_model_name, best_model_instance, best_params, best_score)
        best_score 可能是单个浮点数或字典（如果使用多指标评分）。
    """
    if cv_folds is None:
        cv_folds = config.DEFAULT_CV_FOLDS
    
    results = []
    
    # 确保 y_train 是 NumPy 数组，特别是对于多输出情况
    if isinstance(y_train, pd.DataFrame):
        _y_train = y_train.to_numpy()
    elif isinstance(y_train, pd.Series): # 单目标
        _y_train = y_train.to_numpy().reshape(-1, 1)
    else:
        _y_train = y_train # 假设已经是numpy数组

    if _y_train.ndim == 1: # 确保y是二维的 (n_samples, n_targets)
        _y_train = _y_train.reshape(-1, 1)
        
    print(f"\n--- 开始AutoML: 模型选择与超参数优化 (CV={cv_folds}, 搜索策略={'RandomizedSearch' if use_random_search else 'GridSearch'}) ---")
    print(f"评分标准: {scoring}")
    if isinstance(scoring, dict) and refit_metric:
        print(f"Refit 指标: {refit_metric}")

    for model_name, (model_getter, param_grid) in models_param_grids.items():
        print(f"\n正在优化模型: {model_name}...")
        
        # 获取未训练的模型实例
        # 注意：这里不传递参数给getter，因为参数将在搜索中被设置
        # 但如果getter本身需要一些固定参数，则需要调整
        model_instance = model_getter() 
        
        # SVR等模型可能被MultiOutputRegressor包装，其参数需要 'estimator__' 前缀
        # 检查模型是否是MultiOutputRegressor，如果是，则调整参数网格的键
        # 这部分逻辑可以更通用化，但这里针对SVR做一个简单处理
        current_param_grid = param_grid.copy()
        if isinstance(model_instance, model_provider.MultiOutputRegressor) and model_name == "SVR": # 假设SVR总是被包装
            current_param_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
            print(f"  检测到 {model_name} 是MultiOutputRegressor，参数网格已调整。")


        if use_random_search and current_param_grid: # 仅当有参数可供搜索时使用RandomizedSearch
            search_cv = RandomizedSearchCV(
                estimator=model_instance,
                param_distributions=current_param_grid,
                n_iter=n_iter_random,
                cv=cv_folds,
                scoring=scoring,
                random_state=config.RANDOM_STATE,
                n_jobs=-1, # 使用所有核心
                verbose=1,
                refit=refit_metric if isinstance(scoring, dict) else True # refit基于主要指标
            )
        elif current_param_grid: # 使用GridSearchCV
             search_cv = GridSearchCV(
                estimator=model_instance,
                param_grid=current_param_grid,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                refit=refit_metric if isinstance(scoring, dict) else True
            )
        else: # 如果没有参数网格 (例如，对于某些固定参数的模型如LR的默认情况)
            print(f"  模型 {model_name} 没有提供参数网格，将使用默认参数进行评估。")
            # 简单地训练和评估，或者跳过（取决于需求）
            # 这里我们选择跳过优化步骤，但可以修改为直接评估默认模型
            results.append({
                'model_name': model_name,
                'best_estimator': model_instance.fit(X_train, _y_train), # 训练默认模型
                'best_params': {},
                'best_score': model_instance.score(X_train, _y_train) if hasattr(model_instance, 'score') else -np.inf, # 简单用训练集R2
                'cv_results': None
            })
            print(f"  模型 {model_name} 使用默认参数。")
            continue


        try:
            print(f"  开始对 {model_name} 进行参数搜索...")
            search_cv.fit(X_train, _y_train) # X_train可以是DataFrame, _y_train是NumPy数组
            
            # best_score_ 对于多指标评分，如果refit=True，则返回refit指标的分数
            # 如果refit是字符串，则返回该指标的分数
            # cv_results_ 是一个字典，包含了所有折叠和参数组合的详细结果
            results.append({
                'model_name': model_name,
                'best_estimator': search_cv.best_estimator_,
                'best_params': search_cv.best_params_,
                'best_score': search_cv.best_score_, 
                'cv_results': search_cv.cv_results_
            })
            print(f"  {model_name} 优化完成。最佳分数: {search_cv.best_score_:.4f}")
            print(f"  最佳参数: {search_cv.best_params_}")
        except Exception as e:
            print(f"  优化模型 {model_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'model_name': model_name,
                'best_estimator': None,
                'best_params': None,
                'best_score': -np.inf if not isinstance(scoring, dict) else {k: -np.inf for k in scoring}, # 标记优化失败
                'cv_results': None
            })

    if not results:
        print("没有模型被成功优化。")
        return None, None, None, -np.inf

    # 选择总体最佳模型
    # 如果scoring是单一指标，best_score是数字，越高越好 (例如 R^2)
    # 如果scoring是字典，我们需要基于 refit_metric (如果提供) 或主要指标来比较
    # 这里简化：如果best_score是字典，我们假设用户会自行解读或指定一个主要指标
    # 为了自动选择，如果best_score是字典，我们尝试用refit_metric对应的分数，否则取第一个指标的分数
    
    # 确定用于比较的指标值
    def get_comparison_score(result_item):
        score = result_item['best_score']
        if isinstance(score, dict):
            if refit_metric and refit_metric in score:
                return score[refit_metric]
            # 如果没有refit_metric或不在score中，尝试取第一个指标的值
            # 注意：这可能不总是合适的，取决于指标的性质（越高越好 vs 越低越好）
            # 假设所有指标都是越高越好，或者scoring字典中的顺序是重要的
            return next(iter(score.values())) if score else -np.inf 
        return score if score is not None else -np.inf


    try:
        # 确保所有 'best_score' 都是可比较的 (处理None的情况)
        valid_results = [r for r in results if r['best_estimator'] is not None]
        if not valid_results:
            print("所有模型的优化都失败了。")
            return None, None, None, -np.inf

        best_overall = max(valid_results, key=get_comparison_score)
    except Exception as e:
        print(f"选择最佳模型时发生错误: {e}. 可能是由于评分不一致或None值导致。")
        return None, None, None, -np.inf
    
    print("\n--- AutoML 总结 ---")
    print(f"表现最佳的模型: {best_overall['model_name']}")
    print(f"最佳分数 (基于主要/refit指标): {get_comparison_score(best_overall):.4f}")
    if isinstance(best_overall['best_score'], dict):
        print(f"所有指标的详细分数: {best_overall['best_score']}")
    print(f"最佳参数: {best_overall['best_params']}")
    
    return best_overall['model_name'], best_overall['best_estimator'], best_overall['best_params'], best_overall['best_score']

if __name__ == '__main__':
    from src.data_processing.loader import get_processed_data
    
    print("--- 测试AutoML优化模块 ---")
    try:
        X_train_data, y_train_data, _, _ , _= get_processed_data(scale_features=True) # AutoML通常受益于标准化数据

        if X_train_data.empty or y_train_data.empty:
            raise ValueError("测试AutoML时，数据加载失败。")

        # 定义模型获取函数和参数网格
        # 使用较小的参数空间和迭代次数以加快测试
        models_grids = {
            'LR': (model_provider.get_lr_model, {}), # LR通常没有太多可调超参
            'RF': (model_provider.get_rf_model, {
                'n_estimators': [20, 50], # 减少数量以加快测试
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 4]
            }),
            'XGBR': (model_provider.get_xgbr_model, {
                'n_estimators': [20, 50], 
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1]
            }),
            # SVR的参数网格键名不需要 'estimator__' 前缀，因为我们在函数内部处理了
            # 'SVR': (model_provider.get_svr_model, { 
            #     'C': [0.1, 1.0],
            #     'epsilon': [0.1, 0.2],
            #     'kernel': ['rbf'] # 仅测试rbf以加快速度
            # }) # SVR 训练较慢，可以先注释掉以快速测试其他部分
        }
        
        # 定义评分标准
        # 单一指标示例
        # scoring_metric = 'r2' 
        scoring_metric = 'neg_mean_squared_error' # MSE越小越好，所以用负值，越大越好

        # 多指标评分示例 (可选)
        # scoring_metrics_dict = {
        #     'r2': make_scorer(r2_score, multioutput='uniform_average'),
        #     'neg_mse': make_scorer(mean_squared_error, greater_is_better=False, multioutput='uniform_average')
        # }
        # refit_for_multi = 'r2' # 指定用R2来选择最终模型

        print(f"\n--- 开始AutoML测试 (使用 {scoring_metric} 评分) ---")
        best_name, best_model_instance, best_params_found, best_score_achieved = \
            select_best_model_with_hyperparam_tuning(
                X_train_data, 
                y_train_data, # y_train_data 是 (n_samples, n_targets) 的 DataFrame
                models_grids,
                scoring=scoring_metric, # 或者 scoring_metrics_dict
                n_iter_random=2, # 非常小的迭代次数，仅用于快速测试
                use_random_search=True, # 可以改为False测试GridSearch
                # refit_metric=refit_for_multi if isinstance(scoring_metric, dict) else None
            )

        if best_model_instance:
            print(f"\nAutoML找到的整体最佳模型: {best_name}")
            print(f"其最佳分数 ({scoring_metric if isinstance(scoring_metric, str) else '综合'}): {best_score_achieved}")
            print(f"其最佳参数: {best_params_found}")
            print(f"模型实例: {best_model_instance}")
        else:
            print("AutoML未能找到最佳模型。")

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except ValueError as e:
        print(f"数值错误: {e}")
    except Exception as e:
        print(f"测试AutoML优化模块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()

