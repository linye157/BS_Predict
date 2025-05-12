# src/automl/packaging.py
import joblib
import os
import datetime
from typing import Any

from src import config # 导入配置文件

def package_model(model: Any, model_name: str, version: str = None, subfolder: str = None) -> str | None:
    """
    将训练好的模型打包 (序列化) 到文件。
    文件名将包含模型名称、版本（如果提供）和时间戳。

    参数:
        model (Any): 训练好的模型实例。
        model_name (str): 模型的基础名称 (例如, 'random_forest', 'stacked_model')。
        version (str, optional): 模型版本号。如果为None，则不包含版本。
        subfolder (str, optional): 在 MODEL_OUTPUT_DIR 下创建的子文件夹名称。

    返回:
        str | None: 保存模型的完整路径。如果打包失败，则返回None。
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = [model_name]
    if version:
        filename_parts.append(f"v{version}")
    filename_parts.append(timestamp)
    
    filename = "_".join(filename_parts) + ".joblib"
    
    output_directory = config.MODEL_OUTPUT_DIR
    if subfolder:
        output_directory = os.path.join(config.MODEL_OUTPUT_DIR, subfolder)
        os.makedirs(output_directory, exist_ok=True) # 确保子文件夹存在

    filepath = os.path.join(output_directory, filename)
    
    try:
        print(f"正在将模型 '{model_name}' 打包到: {filepath} ...")
        joblib.dump(model, filepath)
        print(f"模型 '{model_name}' (版本: {version or 'N/A'}, 时间戳: {timestamp}) 已成功打包。")
        return filepath
    except Exception as e:
        print(f"打包模型 {model_name} 时发生错误: {e}")
        return None

def load_packaged_model(filepath: str) -> Any | None:
    """
    从指定的文件路径加载打包好的模型。

    参数:
        filepath (str): 模型的完整文件路径。

    返回:
        Any | None: 加载的模型实例。如果加载失败或文件不存在，则返回None。
    """
    if not os.path.exists(filepath):
        print(f"错误: 模型文件未在路径 {filepath} 找到。")
        return None
        
    try:
        print(f"正在从路径: {filepath} 加载模型...")
        model = joblib.load(filepath)
        print(f"模型已成功从 {filepath} 加载。")
        return model
    except Exception as e:
        print(f"加载模型 {filepath} 时发生错误: {e}")
        return None

def find_latest_model(model_name_pattern: str, subfolder: str = None) -> str | None:
    """
    在模型输出目录中查找符合特定名称模式的最新模型文件。
    文件名通常包含时间戳，此函数利用时间戳排序。

    参数:
        model_name_pattern (str): 模型名称的模式，例如 "random_forest" 或 "automl_best_RF"。
                                  函数会查找包含此模式的文件。
        subfolder (str, optional): 模型所在的子文件夹。

    返回:
        str | None: 最新模型的完整路径。如果找不到匹配的模型，则返回None。
    """
    search_directory = config.MODEL_OUTPUT_DIR
    if subfolder:
        search_directory = os.path.join(config.MODEL_OUTPUT_DIR, subfolder)

    if not os.path.isdir(search_directory):
        print(f"模型目录 {search_directory} 不存在。")
        return None

    candidate_files = []
    for f_name in os.listdir(search_directory):
        if model_name_pattern in f_name and f_name.endswith(".joblib"):
            # 尝试从文件名中解析时间戳 (假设格式如 model_version_YYYYMMDD_HHMMSS.joblib)
            try:
                # 提取文件名中的时间戳部分
                timestamp_str = f_name.split('_')[-1].split('.')[0] # YYYYMMDD
                if len(f_name.split('_')) > 2 and len(f_name.split('_')[-2]) > 6 : # YYYYMMDD_HHMMSS
                     timestamp_str = "_".join(f_name.split('_')[-2:]).split('.')[0]
                
                file_timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                candidate_files.append((file_timestamp, os.path.join(search_directory, f_name)))
            except (ValueError, IndexError) as e:
                # print(f"警告: 无法从文件名 {f_name} 解析时间戳: {e}。将使用文件修改时间。")
                # 如果无法从文件名解析，可以使用文件的最后修改时间作为备选
                try:
                    mtime = os.path.getmtime(os.path.join(search_directory, f_name))
                    candidate_files.append((datetime.datetime.fromtimestamp(mtime), os.path.join(search_directory, f_name)))
                except Exception as ex_mtime:
                    print(f"获取文件 {f_name} 修改时间失败: {ex_mtime}")


    if not candidate_files:
        print(f"在目录 {search_directory} 中未找到名称模式为 '{model_name_pattern}' 的模型文件。")
        return None

    # 按时间戳降序排序，最新的在最前面
    candidate_files.sort(key=lambda x: x[0], reverse=True)
    latest_model_path = candidate_files[0][1]
    print(f"找到最新的 '{model_name_pattern}' 模型: {latest_model_path} (时间戳: {candidate_files[0][0]})")
    return latest_model_path


if __name__ == '__main__':
    from src.models.model_provider import get_lr_model
    from src.data_processing.loader import get_processed_data # 用于简单训练一个模型

    print("--- 测试模型打包与加载模块 ---")
    try:
        # 准备一个简单模型用于测试
        X_train_sample, y_train_sample, _, _, _ = get_processed_data(scale_features=False)
        
        if not X_train_sample.empty and not y_train_sample.empty:
            print("\n训练一个简单的LR模型用于打包测试...")
            lr_model_instance = get_lr_model()
            # y_train_sample 可能有多列，LR原生支持
            lr_model_instance.fit(X_train_sample, y_train_sample) 
            print("简单LR模型训练完毕。")

            # 1. 打包模型
            model_basename = "linear_regression_test"
            model_version = "0.1"
            model_subfolder = "test_models" # 将测试模型放在子文件夹中

            saved_model_path = package_model(
                lr_model_instance, 
                model_basename, 
                version=model_version,
                subfolder=model_subfolder
            )

            if saved_model_path:
                print(f"模型已保存到: {saved_model_path}")

                # 2. 直接通过路径加载模型
                print("\n通过完整路径加载模型...")
                loaded_lr_direct = load_packaged_model(saved_model_path)
                if loaded_lr_direct:
                    print("模型通过路径加载成功。")
                    # 可以尝试用加载的模型进行预测
                    sample_pred = loaded_lr_direct.predict(X_train_sample.head())
                    print("加载模型的样本预测 (前5条训练数据):\n", sample_pred)
                else:
                    print("通过路径加载模型失败。")

                # 3. 查找并加载最新模型
                print("\n查找并加载最新的测试模型...")
                # 使用基础名称和子文件夹来查找
                latest_path = find_latest_model(model_basename, subfolder=model_subfolder) 
                if latest_path:
                    loaded_lr_latest = load_packaged_model(latest_path)
                    if loaded_lr_latest:
                        print("最新的模型加载成功。")
                    else:
                        print("加载最新模型失败。")
                else:
                    print("未找到最新的测试模型。")
            else:
                print("模型打包失败，无法进行加载测试。")
        else:
            print("由于数据加载问题，跳过打包模块的详细测试。")
            
    except FileNotFoundError as e:
        print(f"文件未找到错误: {e}")
    except Exception as e:
        print(f"测试模型打包模块时发生未知错误: {e}")
        import traceback
        traceback.print_exc()
