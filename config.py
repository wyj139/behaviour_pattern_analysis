# 配置文件：项目全局可调参数
from typing import List

# ==================== 数据路径 ====================
# 输入文件路径（可在运行时通过参数覆盖）
RAW_META_CSV: str = "dlc_with_housing.csv"
RAW_TIMESERIES_CSV: str = "train_data.csv"
PREPROCESSED_DATA_DIR: str = "prepared_data"
OUTPUT_BASE_DIR: str = "prepared_data/1+all_back_point"

# ==================== UID 生成配置 ====================
UID_COLS: List[str] = ["day", "pig", "trial"]
UID_COL_NAME: str = "id"
UID_SEP: str = "-"

# ==================== 数据预处理参数 ====================
SMOOTH_MAX_SPEED: float = 5.0  # 每帧最大移动速度
INTERPOLATE_LIMIT_DIRECTION: str = "both"
FILLNA_METHODS = ("ffill", "bfill")
PAD_VALUE_IF_ALL_NA: float = 0.0
LIKELIHOOD_THRESHOLD: float = 0.2

# ==================== 窗口参数 ====================
WINDOW_SIZE: int = 50  # 窗口大小（帧数）
WINDOW_STRIDE: int = 20  # 滑动步长（帧数）

# ==================== Autoencoder 参数 ====================
EMBEDDING_DIM: int = 32  # 嵌入维度
CONV_CHANNELS: int = 32  # 卷积通道数
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 64
NUM_EPOCHS: int = 360
TRAIN_TEST_SPLIT: float = 0.2  # 测试集比例
RANDOM_SEED: int = 42

# ==================== HMM 参数 ====================
N_STATES: int = 15  # HMM 状态数（可自由修改）
HMM_COVARIANCE_TYPE: str = "diag"  # 协方差类型
HMM_N_ITER: int = 200  # HMM 训练迭代次数

# ==================== UMAP 参数 ====================
UMAP_N_COMPONENTS: int = 2
UMAP_RANDOM_STATE: int = 42
UMAP_COLOR_OPTIONS: List[str] = ["pig", "cue", "Housing", "state", "decision"]  # 可用的染色条件

# ==================== 可视化跳过选项 ====================
SKIP_VIS_EMBEDDING_UMAP: bool = True  # 是否跳过基于embedding的多个UMAP图
SKIP_VIS_COORDINATE_UMAP: bool = True  # 是否跳过基于原始坐标的UMAP图
SKIP_VIS_WINDOW_EMBEDDING_UMAP: bool = False  # 是否跳过第一窗口embedding UMAP
SKIP_VIS_WINDOW_COORDINATE_UMAP: bool = False  # 是否跳过第一窗口坐标UMAP

# ==================== 其他 ====================
VERBOSE: bool = True
DEFAULT_FILTERS = []