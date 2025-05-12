# src/config.py
import os

# 项目根目录的智能检测
# BASE_DIR 为项目根文件夹
# __file__ 是当前文件 (config.py) 的路径
# os.path.abspath(__file__) 获取 config.py 的绝对路径
# os.path.dirname() 连续两次用于向上两级目录，即从 src/config.py 到 analysis_diagnosis_system/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径配置
DATA_DIR = os.path.join(BASE_DIR, 'data')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_data.xlsx')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'test_data.xlsx')

# 模型输出目录配置
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'trained_models')
# 确保模型输出目录存在
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# 特征和目标列定义
# 根据描述：65个工艺参数特征 ([:-3]) 和 3个目标列 ([-3:])
NUM_FEATURES = 65
NUM_TARGETS = 3
FEATURE_SLICE = slice(None, NUM_FEATURES) # 或者 slice(None, -NUM_TARGETS)
TARGET_SLICE = slice(NUM_FEATURES, None) # 或者 slice(-NUM_TARGETS, None)

# 目标列的建议名称 (用于绘图和报告，如果数据中没有明确的列名)
# 您可以根据实际情况修改这些名称
TARGET_COL_NAMES = [f'目标_{i+1}' for i in range(NUM_TARGETS)]

# 交叉验证默认折数
DEFAULT_CV_FOLDS = 5

# 全局随机种子，用于保证实验的可复现性
RANDOM_STATE = 42

# 部分模型的默认参数 (可以根据需要进行扩展或修改)
RF_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1 # 使用所有可用的CPU核心
}

XGBR_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror', # 回归任务的平方误差损失
    'n_jobs': -1 # 使用所有可用的CPU核心
}

SVR_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'epsilon': 0.1
    # SVR 对于多输出需要使用 MultiOutputRegressor 进行包装
}

ANN_PARAMS = {
    'hidden_layer_sizes': (100,), # 一个包含100个神经元的隐藏层
    'activation': 'relu',
    'solver': 'adam',
    'max_iter': 500,
    'random_state': RANDOM_STATE,
    'early_stopping': True, # 启用早停以防止过拟合
    'n_iter_no_change': 10 # 早停的耐心值
}

print(f"配置模块加载完毕。项目根目录: {BASE_DIR}")
print(f"训练数据路径: {TRAIN_DATA_PATH}")
print(f"测试数据路径: {TEST_DATA_PATH}")
print(f"模型输出目录: {MODEL_OUTPUT_DIR}")

