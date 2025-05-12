# src/data_processing/loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # 引入数据标准化
from src import config # 从 src 包中导入 config 模块

def load_data(file_path: str) -> pd.DataFrame:
    """
    从Excel文件加载数据。

    参数:
        file_path (str): Excel文件的路径。

    返回:
        pd.DataFrame: 加载的数据帧。如果文件未找到或为空，则返回空的DataFrame。
    """
    try:
        df = pd.read_excel(file_path)
        print(f"数据成功从 {file_path} 加载。形状: {df.shape}")
        if df.empty:
            print(f"警告: 文件 {file_path} 为空。")
        return df
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到。")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载数据时发生错误 {file_path}: {e}")
        return pd.DataFrame()

def split_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    将数据帧分割为特征 (X) 和目标 (y)。

    参数:
        df (pd.DataFrame): 包含特征和目标的数据帧。

    返回:
        tuple[pd.DataFrame, pd.DataFrame]: 特征DataFrame (X) 和目标DataFrame (y)。
                                           如果输入df为空或列数不足，则返回空的DataFrames。
    """
    if df.empty:
        print("错误: 输入的DataFrame为空，无法分割特征和目标。")
        return pd.DataFrame(), pd.DataFrame()

    num_total_columns = df.shape[1]
    expected_min_columns = config.NUM_FEATURES + config.NUM_TARGETS

    if num_total_columns < expected_min_columns:
        print(f"错误: 数据列数 ({num_total_columns}) 少于预期的特征数 ({config.NUM_FEATURES}) + 目标数 ({config.NUM_TARGETS}) = {expected_min_columns}。")
        return pd.DataFrame(), pd.DataFrame()
    
    # 根据config中的切片定义来分割特征和目标
    features = df.iloc[:, config.FEATURE_SLICE]
    targets = df.iloc[:, config.TARGET_SLICE]

    # 为目标列指定预定义的名称 (如果需要)
    if len(config.TARGET_COL_NAMES) == targets.shape[1]:
        targets.columns = config.TARGET_COL_NAMES
    else:
        print(f"警告: config.TARGET_COL_NAMES 长度 ({len(config.TARGET_COL_NAMES)}) 与实际目标列数 ({targets.shape[1]}) 不匹配。将使用默认列名。")


    print(f"特征形状: {features.shape}, 目标形状: {targets.shape}")
    return features, targets

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame, scale_features: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    对特征数据进行预处理，例如标准化。

    参数:
        X_train (pd.DataFrame): 训练集特征。
        X_test (pd.DataFrame): 测试集特征。
        scale_features (bool): 是否执行特征标准化。默认为True。

    返回:
        tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
            处理后的训练集特征, 处理后的测试集特征, 如果进行了标准化则返回StandardScaler对象，否则返回None。
    """
    print("开始数据预处理...")
    scaler = None
    if scale_features:
        print("对特征进行标准化处理...")
        scaler = StandardScaler()
        # 在训练集上拟合并转换
        X_train_scaled = scaler.fit_transform(X_train)
        # 在测试集上仅转换
        X_test_scaled = scaler.transform(X_test)
        
        # 转换回DataFrame并保留列名
        X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        print("特征标准化完成。")
    else:
        print("未执行特征标准化。")
        
    # 此处可以添加其他预处理步骤，例如缺失值处理、编码等
    # 例如，检查缺失值
    if X_train.isnull().sum().sum() > 0:
        print("警告: 训练数据中检测到缺失值。建议进行处理。")
        # X_train = X_train.fillna(X_train.mean()) # 简单示例：用均值填充
    if X_test.isnull().sum().sum() > 0:
        print("警告: 测试数据中检测到缺失值。建议进行处理。")
        # X_test = X_test.fillna(X_test.mean()) # 简单示例：用均值填充

    return X_train, X_test, scaler


def get_processed_data(scale_features: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    """
    加载、分割并预处理训练和测试数据。

    参数:
        scale_features (bool): 是否对特征进行标准化。

    返回:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler | None]:
            X_train, y_train, X_test, y_test, scaler (如果进行了标准化)
            如果数据加载失败，则返回空的DataFrames和None。
    """
    print("开始加载和处理数据...")
    train_df = load_data(config.TRAIN_DATA_PATH)
    test_df = load_data(config.TEST_DATA_PATH)

    if train_df.empty or test_df.empty:
        print("错误: 训练数据或测试数据加载失败。请检查文件路径和内容。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    X_train, y_train = split_features_targets(train_df)
    X_test, y_test = split_features_targets(test_df)

    if X_train.empty or y_train.empty or X_test.empty or y_test.empty:
        print("错误: 特征和目标分割失败。")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    X_train_processed, X_test_processed, scaler = preprocess_data(X_train, X_test, scale_features=scale_features)
    
    print("数据加载和处理完成。")
    return X_train_processed, y_train, X_test_processed, y_test, scaler

if __name__ == '__main__':
    # 测试数据加载和预处理流程
    print("--- 测试数据加载模块 ---")
    X_train_data, y_train_data, X_test_data, y_test_data, fitted_scaler = get_processed_data(scale_features=True)

    if not X_train_data.empty:
        print("\n--- 处理后的训练数据概览 ---")
        print("X_train_data 头部:\n", X_train_data.head())
        print("y_train_data 头部:\n", y_train_data.head())
        if fitted_scaler:
            print(f"StandardScaler 均值 (部分): {fitted_scaler.mean_[:5]}") # 打印部分均值
            print(f"StandardScaler 方差 (部分): {fitted_scaler.var_[:5]}")   # 打印部分方差

        print("\n--- 处理后的测试数据概览 ---")
        print("X_test_data 头部:\n", X_test_data.head())
        print("y_test_data 头部:\n", y_test_data.head())
    else:
        print("数据处理失败，请检查日志。")
