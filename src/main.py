# src/main.py
import pandas as pd
import os
import datetime

# 导入项目模块
from src.data_processing import loader
from src.models import model_provider
from src.ensemble import stacking
from src.automl import optimization, packaging
from src.evaluation import analysis
from src import config as app_config # 使用别名以避免与全局变量config冲突

def create_run_output_folder(base_folder_name: str = "run_results") -> str:
    """为单次运行创建一个带时间戳的输出文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder_name = f"{base_folder_name}_{timestamp}"
    run_output_path = os.path.join(app_config.MODEL_OUTPUT_DIR, "..", "run_outputs", run_folder_name) # 存放在 trained_models 同级的 run_outputs 下
    os.makedirs(run_output_path, exist_ok=True)
    print(f"本次运行的输出将保存在: {run_output_path}")
    return run_output_path


def run_single_model_training_and_evaluation(
    model_name_key: str, # 例如 "RF", "LR"
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
    model_params: dict = None, # 特定模型的参数
    output_folder: str = None # 用于保存图表和模型的文件夹
):
    """
    训练、评估单个指定模型，并可选择打包。
    """
    print(f"\n--- 开始训练和评估单个模型: {model_name_key} ---")
    
    if model_name_key not in model_provider.MODEL_GETTERS:
        print(f"错误: 模型 '{model_name_key}' 未在 model_provider 中定义。跳过此模型。")
        return None, None

    # 1. 获取模型实例
    model_instance = model_provider.MODEL_GETTERS[model_name_key](params=model_params)
    model_display_name = model_instance.__class__.__name__ # 获取实际的模型类名用于显示
    if isinstance(model_instance, model_provider.MultiOutputRegressor): # 特殊处理包装器
        model_display_name = f"MultiOutput({model_instance.estimator.__class__.__name__})"


    # 2. 训练模型
    print(f"正在训练 {model_display_name}...")
    try:
        model_instance.fit(X_train, y_train) # y_train 可以是DataFrame
        print(f"{model_display_name} 训练完成。")
    except Exception as e:
        print(f"训练 {model_display_name} 时发生错误: {e}")
        return None, None

    # 3. 在测试集上进行预测
    y_pred_test = model_instance.predict(X_test)
    # 确保预测结果是DataFrame，以便与y_test (DataFrame) 的列名和索引保持一致
    y_pred_test_df = pd.DataFrame(y_pred_test, columns=y_test.columns, index=y_test.index)

    # 4. 评估模型
    print(f"\n{model_display_name} - 测试集评估指标:")
    test_metrics_df = analysis.calculate_regression_metrics(y_test, y_pred_test_df, target_names=app_config.TARGET_COL_NAMES)

    # 5. 可视化 (为每个目标绘制图表)
    if output_folder:
        num_targets_to_plot = min(y_test.shape[1], len(app_config.TARGET_COL_NAMES))
        for i in range(num_targets_to_plot):
            target_name = app_config.TARGET_COL_NAMES[i]
            analysis.plot_predictions_vs_actual(
                y_test, y_pred_test_df, target_idx=i, target_name=target_name, model_name=model_display_name,
                save_path=os.path.join(output_folder, f"single_{model_name_key}_pred_vs_actual_{target_name}.png")
            )
            analysis.plot_residuals_distribution(
                y_test, y_pred_test_df, target_idx=i, target_name=target_name, model_name=model_display_name,
                save_path=os.path.join(output_folder, f"single_{model_name_key}_residuals_dist_{target_name}.png")
            )
        
        # 绘制特征重要性图 (如果模型支持)
        # 对于MultiOutputRegressor包装的SVR等，需要访问其estimator的coef_或feature_importances_
        # analysis.plot_feature_importance 内部已有一些处理逻辑
        analysis.plot_feature_importance(
            model_instance, X_train.columns.tolist(), model_name=model_display_name, top_n=15,
            save_path=os.path.join(output_folder, f"single_{model_name_key}_feature_importance.png")
        )

    # 6. 打包模型 (可选，但推荐)
    if output_folder: # 仅当提供了输出文件夹时打包
        package_subfolder = os.path.join(os.path.basename(output_folder), "models") # 在运行文件夹下创建models子目录
        packaging.package_model(model_instance, f"single_{model_name_key}", subfolder=package_subfolder)
    
    return model_instance, test_metrics_df


def run_stacking_ensemble_training_and_evaluation(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
    output_folder: str = None
):
    """
    训练、评估Stacking集成模型，并可选择打包。
    """
    print("\n" + "="*60)
    print("--- 开始训练和评估 Stacking 集成模型 ---")
    print("="*60)

    # 1. 定义一级学习器 (Level-1 models) 配置
    # 使用较少的估计器和深度以加快演示速度
    base_learner_configs = [
        ('RF', {'n_estimators': 30, 'max_depth': 7, 'random_state': app_config.RANDOM_STATE}),
        ('XGBR', {'n_estimators': 30, 'max_depth': 4, 'random_state': app_config.RANDOM_STATE}),
        # ('SVR', {'C': 0.8, 'kernel': 'rbf'}) # SVR可能较慢，可以根据需要添加
        ('LR', None) # 也可加入简单模型作为基学习器
    ]

    # 2. 定义元学习器 (Level-2 model) 配置
    meta_learner_config = ('LR', None) # 例如，使用线性回归作为元学习器

    # 3. 训练Stacking模型
    # y_train 已经是DataFrame (n_samples, n_targets)
    # StackingRegressor可以处理多输出的y，只要基础学习器和元学习器也能处理
    # 或者MultiOutputRegressor被用于不支持多输出的基础学习器
    try:
        stacked_model = stacking.train_stacking_model_custom(
            X_train, y_train, 
            base_learner_configs, 
            meta_learner_config,
            cv_folds=3, # 使用较少的折数加速演示
            use_sklearn_stacking=True # 推荐使用内置的StackingRegressor
        )
    except Exception as e:
        print(f"训练Stacking模型时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None
        
    model_display_name = "StackingEnsemble"

    # 4. 在测试集上评估
    y_pred_test_stacked = stacked_model.predict(X_test)
    y_pred_test_stacked_df = pd.DataFrame(y_pred_test_stacked, columns=y_test.columns, index=y_test.index)

    print(f"\n{model_display_name} - 测试集评估指标:")
    test_metrics_df = analysis.calculate_regression_metrics(y_test, y_pred_test_stacked_df, target_names=app_config.TARGET_COL_NAMES)

    # 5. 可视化
    if output_folder:
        num_targets_to_plot = min(y_test.shape[1], len(app_config.TARGET_COL_NAMES))
        for i in range(num_targets_to_plot):
            target_name = app_config.TARGET_COL_NAMES[i]
            analysis.plot_predictions_vs_actual(
                y_test, y_pred_test_stacked_df, target_idx=i, target_name=target_name, model_name=model_display_name,
                save_path=os.path.join(output_folder, f"stacking_pred_vs_actual_{target_name}.png")
            )
        
        # StackingRegressor 通常没有直接的 feature_importances_。
        # 可以查看元学习器的 coef_ (如果它是线性的) 来了解基学习器预测的权重。
        # feature_names_for_meta 需要根据基学习器和是否passthrough来构造
        if hasattr(stacked_model, 'final_estimator_'):
            meta_feature_names = [name for name, _ in base_learner_configs] # 简化版，假设无passthrough且每个基模型单输出
            if stacked_model.passthrough: # 如果原始特征也传给了元学习器
                 meta_feature_names = X_train.columns.tolist() + meta_feature_names * y_train.shape[1] # 粗略估计
            
            # 如果元学习器是多输出的，其coef_可能是二维的。
            # plot_feature_importance会尝试处理这种情况。
            # 注意：这里的meta_feature_names可能不完全准确，取决于StackingRegressor内部如何处理多输出基学习器。
            # 一个更准确的方法是检查 stacked_model.named_estimators_ 和 stacked_model.stack_method_
            analysis.plot_feature_importance(
                stacked_model, # 传递整个stacker，让绘图函数尝试解析
                meta_feature_names, # 这个名字列表可能需要更精确的构建
                model_name=f"{model_display_name} (Meta-Learner)", top_n=15,
                save_path=os.path.join(output_folder, "stacking_meta_feature_importance.png")
            )

    # 6. 打包模型
    if output_folder:
        package_subfolder = os.path.join(os.path.basename(output_folder), "models")
        packaging.package_model(stacked_model, "stacking_ensemble", subfolder=package_subfolder)
        
    return stacked_model, test_metrics_df


def run_automl_pipeline(
    X_train: pd.DataFrame, 
    y_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_test: pd.DataFrame,
    output_folder: str = None
):
    """
    运行AutoML流程：模型筛选、超参数优化、评估和打包。
    """
    print("\n" + "="*60)
    print("--- 开始运行 AutoML 流程 ---")
    print("="*60)

    # 1. 定义要尝试的模型和它们的参数搜索空间
    # 使用较小的参数空间和迭代次数以加快演示
    models_param_grids_for_automl = {
        'LR': (model_provider.get_lr_model, {}), # LR通常没有太多可调超参
        'RF': (model_provider.get_rf_model, {
            'n_estimators': [20, 50, 80], 
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }),
        'XGBR': (model_provider.get_xgbr_model, {
            'n_estimators': [20, 50, 80], 
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }),
        # SVR的参数网格键名不需要 'estimator__' 前缀，因为 optimization 模块内部会处理
        'SVR': (model_provider.get_svr_model, { 
            'C': [0.1, 1.0, 5.0],
            'kernel': ['rbf', 'linear'], # SVR的kernel选择
            'epsilon': [0.05, 0.1, 0.2]
        }),
        # 'ANN': (model_provider.get_ann_model, { # MLPRegressor
        #     'hidden_layer_sizes': [(50,), (100,), (50,25)],
        #     'alpha': [0.0001, 0.001, 0.01],
        #     'learning_rate_init': [0.001, 0.005, 0.01]
        # }) # ANN 训练可能较慢, 演示时可以注释掉
    }
    
    # 2. 选择评分标准
    # 对于多输出，这些评分器通常会计算每个目标的得分然后平均
    # scorer_for_automl = 'r2' 
    scorer_for_automl = 'neg_mean_squared_error' # MSE越小越好，所以用负值，越大越好

    # 3. 运行AutoML优化
    try:
        best_name, best_model, best_params, best_score = \
            optimization.select_best_model_with_hyperparam_tuning(
                X_train, y_train, 
                models_param_grids_for_automl,
                scoring=scorer_for_automl,
                n_iter_random=5, # 随机搜索的迭代次数 (如果用RandomizedSearchCV)
                                 # 对于GridSearchCV，这个参数会被忽略
                use_random_search=True, # True: RandomizedSearchCV, False: GridSearchCV
                cv_folds=3 # 减少CV折数以加速演示
            )
    except Exception as e:
        print(f"AutoML优化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None

    if not best_model:
        print("AutoML 流程未能成功找到最佳模型。")
        return None, None
        
    model_display_name = f"AutoML_Best ({best_name})"
    print(f"\nAutoML 找到的最佳模型: {model_display_name}")
    print(f"  最佳分数 ({scorer_for_automl}): {best_score}") # best_score可能是字典
    print(f"  最佳参数: {best_params}")

    # 4. 使用找到的最佳模型在测试集上评估
    y_pred_test_automl = best_model.predict(X_test)
    y_pred_test_automl_df = pd.DataFrame(y_pred_test_automl, columns=y_test.columns, index=y_test.index)

    print(f"\n{model_display_name} - 测试集评估指标:")
    test_metrics_df = analysis.calculate_regression_metrics(y_test, y_pred_test_automl_df, target_names=app_config.TARGET_COL_NAMES)

    # 5. 可视化
    if output_folder:
        num_targets_to_plot = min(y_test.shape[1], len(app_config.TARGET_COL_NAMES))
        for i in range(num_targets_to_plot):
            target_name = app_config.TARGET_COL_NAMES[i]
            analysis.plot_predictions_vs_actual(
                y_test, y_pred_test_automl_df, target_idx=i, target_name=target_name, model_name=model_display_name,
                save_path=os.path.join(output_folder, f"automl_{best_name}_pred_vs_actual_{target_name}.png")
            )
        
        analysis.plot_feature_importance(
            best_model, X_train.columns.tolist(), model_name=model_display_name, top_n=15,
            save_path=os.path.join(output_folder, f"automl_{best_name}_feature_importance.png")
        )

    # 6. 打包最佳模型
    if output_folder:
        package_subfolder = os.path.join(os.path.basename(output_folder), "models")
        packaging.package_model(best_model, f"automl_best_{best_name.lower().replace(' ', '_')}", subfolder=package_subfolder)
        
    return best_model, test_metrics_df


def main():
    """
    主执行函数，按顺序调用各个机器学习流程。
    """
    print("开始执行机器学习子系统主流程...")
    
    # 为本次运行创建一个统一的输出文件夹
    current_run_output_folder = create_run_output_folder("ml_subsystem_run")
    print(f"所有图表和模型将尝试保存到: {current_run_output_folder}")

    try:
        # 1. 加载和预处理数据 (系统接口：数据接口)
        # 特征标准化通常对LR, SVR, ANN有益，对树模型影响不大。这里统一进行标准化。
        X_train, y_train, X_test, y_test, scaler = loader.get_processed_data(scale_features=True)
        
        if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
            print("错误: 数据加载或预处理失败。无法继续执行。")
            return

        # --- 演示单个模型训练 (例如, 随机森林 和 SVR) ---
        # 系统接口：参数设置 (通过model_provider中的默认或传递参数)
        # 机器学习：随机森林(RF)模型, 支持向量机(SVR)模型等
        print("\n" + "="*60)
        print("阶段一: 单个模型训练与评估")
        print("="*60)
        
        # 示例1: 随机森林
        run_single_model_training_and_evaluation(
            "RF", X_train, y_train, X_test, y_test, 
            model_params={'n_estimators': 50, 'max_depth': 10}, # 覆盖默认参数以加速
            output_folder=current_run_output_folder
        )
        
        # 示例2: 支持向量机 (SVR)
        # SVR可能训练较慢，特别是对于大数据集
        run_single_model_training_and_evaluation(
            "SVR", X_train, y_train, X_test, y_test,
            model_params={'C': 1.0, 'kernel': 'rbf'}, # SVR的参数
            output_folder=current_run_output_folder
        )
        # 系统接口：训练过程可视化、训练误差分析 (在run_single_model_training_and_evaluation内部调用)


        # --- 演示机器学习Stacking集成 ---
        # 机器学习Stacking集成：训练一级模型、训练二级模型、k折交叉验证
        run_stacking_ensemble_training_and_evaluation(
            X_train, y_train, X_test, y_test,
            output_folder=current_run_output_folder
        )
        # 系统接口：训练误差分析 (在run_stacking_ensemble_training_and_evaluation内部调用)


        # --- 演示自动化机器学习 ---
        # 自动化机器学习：模型自动筛选、模型参数自动最优化设置、模型打包制作与输出
        run_automl_pipeline(
            X_train, y_train, X_test, y_test,
            output_folder=current_run_output_folder
        )
        # 系统接口：参数设置与调优接口 (通过optimization模块实现)
        # 系统接口：训练误差分析 (在run_automl_pipeline内部调用)

        print("\n机器学习子系统主流程执行完毕。")

    except FileNotFoundError as e:
        print(f"主流程中发生文件未找到错误: {e}")
    except ValueError as e:
        print(f"主流程中发生数值或参数错误: {e}")
    except Exception as e:
        print(f"主流程中发生未预料的错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # 设置matplotlib后端，以便在无GUI环境（如某些服务器或Docker容器）中也能保存图像
    # import matplotlib
    # matplotlib.use('Agg') # 'Agg'是一个非交互式后端，图像直接写入文件

    main()
