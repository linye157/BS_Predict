# src/evaluation/analysis.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import List, Dict, Any, Union
import os

from src import config as app_config # 使用别名以避免与函数参数冲突

# 设置matplotlib支持中文显示 (如果需要，选择合适的字体)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 例如: SimHei (黑体)
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题
# 注意：确保系统中安装了所选字体，否则可能引发警告或错误。
# 如果在服务器环境或没有GUI的环境中运行，绘图可能需要特殊处理（例如保存到文件而不是显示）。

def calculate_regression_metrics(
    y_true: Union[pd.DataFrame, np.ndarray], 
    y_pred: Union[pd.DataFrame, np.ndarray], 
    target_names: List[str] = None
) -> pd.DataFrame:
    """
    计算回归模型的评估指标 (MSE, RMSE, MAE, R2)，支持多输出。

    参数:
        y_true (pd.DataFrame or np.ndarray): 真实目标值, 形状 (n_samples, n_targets)。
        y_pred (pd.DataFrame or np.ndarray): 预测目标值, 形状 (n_samples, n_targets)。
        target_names (List[str], optional): 目标列的名称列表。
                                            如果y_true是DataFrame且此参数为None，则尝试使用y_true的列名。
                                            如果仍无法确定，则使用默认名称如 "目标_1"。

    返回:
        pd.DataFrame: 包含每个目标和平均（如果多目标）指标的DataFrame。
    """
    # 统一数据类型为 NumPy 数组
    if isinstance(y_true, pd.DataFrame):
        if target_names is None: # 优先使用DataFrame的列名
            target_names = y_true.columns.tolist()
        y_true_arr = y_true.to_numpy()
    elif isinstance(y_true, np.ndarray):
        y_true_arr = y_true
    else:
        raise TypeError("y_true 必须是 pandas DataFrame 或 NumPy array 类型。")

    if isinstance(y_pred, pd.DataFrame):
        y_pred_arr = y_pred.to_numpy()
    elif isinstance(y_pred, np.ndarray):
        y_pred_arr = y_pred
    else:
        raise TypeError("y_pred 必须是 pandas DataFrame 或 NumPy array 类型。")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(f"形状不匹配: y_true {y_true_arr.shape}, y_pred {y_pred_arr.shape}")

    # 确定目标数量
    if y_true_arr.ndim == 1: # 单目标情况，转换为二维
        y_true_arr = y_true_arr.reshape(-1, 1)
        y_pred_arr = y_pred_arr.reshape(-1, 1)
    
    n_targets = y_true_arr.shape[1]
    
    # 确定目标名称
    if target_names is None or len(target_names) != n_targets:
        if n_targets <= len(app_config.TARGET_COL_NAMES):
             target_names = app_config.TARGET_COL_NAMES[:n_targets]
        else: # 如果实际目标数超过预设名称数
            target_names = [f"目标_{i+1}" for i in range(n_targets)]
        print(f"警告: 未提供有效的目标名称或数量不匹配，已使用默认名称: {target_names}")


    metrics_data = []

    for i in range(n_targets):
        yt_single = y_true_arr[:, i]
        yp_single = y_pred_arr[:, i]
        
        mse = mean_squared_error(yt_single, yp_single)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(yt_single, yp_single)
        r2 = r2_score(yt_single, yp_single)
        
        metrics_data.append({
            '目标': target_names[i],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        })
    
    # 如果是多目标，计算平均指标
    if n_targets > 1:
        avg_mse = mean_squared_error(y_true_arr, y_pred_arr, multioutput='uniform_average')
        avg_rmse = np.sqrt(avg_mse) # 基于平均MSE计算RMSE
        avg_mae = mean_absolute_error(y_true_arr, y_pred_arr, multioutput='uniform_average')
        avg_r2 = r2_score(y_true_arr, y_pred_arr, multioutput='uniform_average')
        
        metrics_data.append({
            '目标': '平均值 (Uniform Avg)', # 指明平均方式
            'MSE': avg_mse,
            'RMSE': avg_rmse,
            'MAE': avg_mae,
            'R2': avg_r2
        })
        
    metrics_df = pd.DataFrame(metrics_data).set_index('目标') # 将“目标”列设为索引
    print("\n--- 回归模型评估指标 ---")
    print(metrics_df)
    return metrics_df

def plot_predictions_vs_actual(
    y_true: Union[pd.DataFrame, np.ndarray], 
    y_pred: Union[pd.DataFrame, np.ndarray], 
    target_idx: int = 0, 
    target_name: str = None, 
    model_name: str = "模型",
    save_path: str = None
):
    """
    绘制单个目标的预测值 vs 真实值散点图。

    参数:
        y_true, y_pred: 真实值和预测值。
        target_idx (int): 要绘制的目标的索引 (从0开始)。
        target_name (str, optional): 目标名称，用于图表标题。
        model_name (str): 模型名称，用于图表标题。
        save_path (str, optional): 如果提供，则将图表保存到此路径，而不是显示。
    """
    # 从 y_true 和 y_pred 中提取指定目标的数据
    if isinstance(y_true, pd.DataFrame):
        yt_col = y_true.iloc[:, target_idx]
        if target_name is None: target_name = y_true.columns[target_idx]
    else: # np.ndarray
        yt_col = y_true[:, target_idx] if y_true.ndim > 1 else y_true
        if target_name is None: target_name = f"目标 {target_idx+1}"
        
    if isinstance(y_pred, pd.DataFrame):
        yp_col = y_pred.iloc[:, target_idx]
    else: # np.ndarray
        yp_col = y_pred[:, target_idx] if y_pred.ndim > 1 else y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(yt_col, yp_col, alpha=0.7, label='预测值', color='royalblue')
    
    # 绘制理想线 (y=x)
    min_val = min(yt_col.min(), yp_col.min())
    max_val = max(yt_col.max(), yp_col.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想情况 (y=x)')
    
    plt.xlabel(f"真实值 - {target_name}")
    plt.ylabel(f"预测值 - {target_name}")
    plt.title(f"{model_name}: 真实值 vs. 预测值 ({target_name})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.close() # 保存后关闭，避免在无GUI环境显示
    else:
        plt.show()

def plot_residuals_distribution(
    y_true: Union[pd.DataFrame, np.ndarray], 
    y_pred: Union[pd.DataFrame, np.ndarray], 
    target_idx: int = 0, 
    target_name: str = None, 
    model_name: str = "模型",
    save_path: str = None
):
    """
    绘制单个目标的残差分布直方图和KDE图。

    参数: (同上)
    """
    if isinstance(y_true, pd.DataFrame):
        yt_col = y_true.iloc[:, target_idx]
        if target_name is None: target_name = y_true.columns[target_idx]
    else:
        yt_col = y_true[:, target_idx] if y_true.ndim > 1 else y_true
        if target_name is None: target_name = f"目标 {target_idx+1}"
        
    if isinstance(y_pred, pd.DataFrame):
        yp_col = y_pred.iloc[:, target_idx]
    else:
        yp_col = y_pred[:, target_idx] if y_pred.ndim > 1 else y_pred
        
    residuals = yt_col - yp_col
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='teal', stat="density") # 使用 stat="density" 使KDE和直方图可比
    plt.axvline(residuals.mean(), color='r', linestyle='dashed', linewidth=1.5, label=f'残差均值: {residuals.mean():.2f}')
    plt.axvline(0, color='k', linestyle='solid', linewidth=1, label='零残差线')
    
    plt.xlabel(f"残差 ({target_name})")
    plt.ylabel("密度") # 如果使用 stat="density"
    plt.title(f"{model_name}: 残差分布 ({target_name})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.close()
    else:
        plt.show()

def plot_residuals_vs_predicted(
    y_true: Union[pd.DataFrame, np.ndarray], 
    y_pred: Union[pd.DataFrame, np.ndarray], 
    target_idx: int = 0, 
    target_name: str = None, 
    model_name: str = "模型",
    save_path: str = None
):
    """
    绘制单个目标的残差 vs. 预测值散点图。

    参数: (同上)
    """
    if isinstance(y_true, pd.DataFrame):
        yt_col = y_true.iloc[:, target_idx]
        if target_name is None: target_name = y_true.columns[target_idx]
    else:
        yt_col = y_true[:, target_idx] if y_true.ndim > 1 else y_true
        if target_name is None: target_name = f"目标 {target_idx+1}"
        
    if isinstance(y_pred, pd.DataFrame):
        yp_col = y_pred.iloc[:, target_idx]
    else:
        yp_col = y_pred[:, target_idx] if y_pred.ndim > 1 else y_pred
        
    residuals = yt_col - yp_col
    
    plt.figure(figsize=(10, 6))
    plt.scatter(yp_col, residuals, alpha=0.6, color='purple')
    plt.axhline(0, color='r', linestyle='--', lw=2, label='零残差线')
    
    plt.xlabel(f"预测值 ({target_name})")
    plt.ylabel(f"残差 ({target_name})")
    plt.title(f"{model_name}: 残差 vs. 预测值 ({target_name})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(
    model: Any, 
    feature_names: List[str], 
    model_name: str = "模型", 
    top_n: int = 20,
    save_path: str = None
):
    """
    绘制特征重要性条形图。
    适用于具有 feature_importances_ (如树模型) 或 coef_ (如线性模型) 属性的模型。
    对于多输出线性模型，会尝试平均各输出的系数绝对值。

    参数:
        model (Any): 训练好的模型实例。
        feature_names (List[str]): 特征名称列表。
        model_name (str): 模型名称，用于图表标题。
        top_n (int): 显示最重要的前 N 个特征。
        save_path (str, optional): 如果提供，则将图表保存到此路径。
    """
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        if model.coef_.ndim > 1: # 多输出线性模型
            # 取各输出目标对应系数的绝对值的平均值作为重要性度量
            importances = np.mean(np.abs(model.coef_), axis=0) 
            print(f"注意: {model_name} 是多输出线性模型，特征重要性是基于系数绝对值的平均值。")
        else: # 单输出线性模型
            importances = np.abs(model.coef_)
    # 对于StackingRegressor的元学习器
    elif hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'coef_'):
        # 这是元学习器的系数，对应于基学习器的预测（以及可能的原始特征）
        # feature_names 此时应对应于元学习器的输入特征名
        if model.final_estimator_.coef_.ndim > 1:
            importances = np.mean(np.abs(model.final_estimator_.coef_), axis=0)
        else:
            importances = np.abs(model.final_estimator_.coef_)
        print(f"注意: {model_name} 是Stacking模型，显示的是元学习器的特征重要性。")
    # 对于被MultiOutputRegressor包装的模型
    elif hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        # 尝试获取第一个estimator的特征重要性 (简化处理)
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, 'feature_importances_'):
            importances = first_estimator.feature_importances_
            print(f"注意: {model_name} 是MultiOutputRegressor包装模型，显示的是第一个子模型的特征重要性。")
        elif hasattr(first_estimator, 'coef_'):
            importances = np.abs(first_estimator.coef_) # 假设单输出
            print(f"注意: {model_name} 是MultiOutputRegressor包装模型，显示的是第一个子模型系数的绝对值。")


    if importances is None:
        print(f"模型 {model_name} (或其相关组件) 不具备 'feature_importances_' 或 'coef_' 属性，无法绘制特征重要性。")
        return

    if len(importances) != len(feature_names):
        print(f"警告: 特征重要性数量 ({len(importances)}) 与特征名称数量 ({len(feature_names)}) 不匹配。无法绘制特征重要性。")
        # 这可能发生在Stacking的元学习器，其输入是基模型的预测
        return


    indices = np.argsort(importances)[::-1] # 获取按重要性降序排列的特征索引
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(12, max(6, min(top_n, len(feature_names)) * 0.4))) # 动态调整高度
    sns.barplot(x=importances[top_indices], y=np.array(feature_names)[top_indices], palette="viridis", orient='h')
    plt.xlabel("相对重要性")
    plt.ylabel("特征")
    plt.title(f"{model_name} - 特征重要性 (Top {min(top_n, len(feature_names))})")
    plt.tight_layout() # 调整布局以防止标签重叠

    if save_path:
        plt.savefig(save_path)
        print(f"图表已保存到: {save_path}")
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    from src.data_processing.loader import get_processed_data
    from src.models.model_provider import get_lr_model # 用于简单测试

    print("--- 测试评估与分析模块 ---")
    try:
        # 加载数据
        X_train_data, y_train_data, X_test_data, y_test_data, _ = get_processed_data(scale_features=True)

        if X_train_data.empty or y_train_data.empty or X_test_data.empty or y_test_data.empty:
            raise ValueError("测试评估模块时，数据加载失败。")

        # 训练一个简单模型用于测试评估函数
        print("\n训练一个简单的LR模型用于评估测试...")
        lr_model = get_lr_model()
        lr_model.fit(X_train_data, y_train_data)
        
        # 在测试集上进行预测
        y_pred_test_lr = lr_model.predict(X_test_data)
        # 将 NumPy 预测转换为 DataFrame，以便与 y_test (DataFrame) 保持一致的列名和索引
        y_pred_test_lr_df = pd.DataFrame(y_pred_test_lr, columns=y_test_data.columns, index=y_test_data.index)

        print("\n--- 在测试集上评估LR模型 ---")
        # 使用 app_config.TARGET_COL_NAMES 作为目标名称
        metrics_df = calculate_regression_metrics(y_test_data, y_pred_test_lr_df, target_names=app_config.TARGET_COL_NAMES)

        # 为每个目标绘制图表 (假设目标数量不超过 app_config.TARGET_COL_NAMES 的长度)
        num_targets_to_plot = min(y_test_data.shape[1], len(app_config.TARGET_COL_NAMES))
        
        # 创建一个目录来保存图表 (可选)
        plots_output_dir = os.path.join(app_config.BASE_DIR, "evaluation_plots_test")
        os.makedirs(plots_output_dir, exist_ok=True)

        for i in range(num_targets_to_plot):
            current_target_name = app_config.TARGET_COL_NAMES[i]
            print(f"\n为 {current_target_name} 生成可视化图表...")

            plot_predictions_vs_actual(
                y_test_data, y_pred_test_lr_df, 
                target_idx=i, target_name=current_target_name, model_name="线性回归",
                save_path=os.path.join(plots_output_dir, f"lr_pred_vs_actual_{current_target_name}.png")
            )
            plot_residuals_distribution(
                y_test_data, y_pred_test_lr_df, 
                target_idx=i, target_name=current_target_name, model_name="线性回归",
                save_path=os.path.join(plots_output_dir, f"lr_residuals_dist_{current_target_name}.png")
            )
            plot_residuals_vs_predicted(
                y_test_data, y_pred_test_lr_df,
                target_idx=i, target_name=current_target_name, model_name="线性回归",
                save_path=os.path.join(plots_output_dir, f"lr_residuals_vs_pred_{current_target_name}.png")
            )
        
        # 绘制特征重要性 (LR有coef_)
        print("\n绘制线性回归模型的特征重要性...")
        plot_feature_importance(
            lr_model, 
            X_train_data.columns.tolist(), 
            model_name="线性回归", 
            top_n=15,
            save_path=os.path.join(plots_output_dir, "lr_feature_importance.png")
        )
        print(f"测试图表已保存到目录: {plots_output_dir}")

    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
    except ValueError as e:
        print(f"数值错误: {e}")
    except Exception as e:
        print(f"测试评估分析模块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()

