from __future__ import annotations

from typing import Tuple

DEBUG = True
# ---- 常温时间-密度模式开关 ----
ROOM_TEMP_TIME_MODE = False #若为true，加载常温下的图像，路径在以下定义
ROOM_TEMP_TIME_USE_ELAPSED_MINUTES = True
ROOM_TEMP_CONSTANT_TEMP_C = None

NAME = 'Cu44'
# ----- get_temp.py 相关默认设置 -----
DEFAULT_CSV = './ESL Data/' + NAME + '/' + NAME + '.csv'

if ROOM_TEMP_TIME_MODE:
    ROOM_TEMP_DIR = './ESL Data/' + NAME + '/' + 'room temperature'
    DEBUG_OUT_DIR = './ESL Data/' + NAME + '/' + 'room temperature' + '/debug_detected'
else:
    ROOM_TEMP_DIR = './ESL Data/' + NAME
    DEBUG_OUT_DIR = './ESL Data/' + NAME + '/debug_detected'

# 像素比与质量文件路径
PIXEL_SCALE_FILE = './ESL Data/' + NAME + '/' + NAME + '_像素比与质量.txt'

# dark_mask_final原始数据保存路径
DARK_MASK_DATA_DIR = './ESL Data/' + NAME + '/dark_mask_data'

PIXEL_RATIO_IS_VOLUME = False

LOW_TEMP_DIR = './ESL Data/' + NAME + '/' + NAME + '_low_temp'

LOW_TEMP = 1085

# 当设置了 LOW_TEMP 并进入低温温度区间，但对应文件不在 LOW_TEMP_DIR 中时的处理策略：
# False (默认旧行为): 跳过该图像，不做体积计算；
# True : 回退使用常温目录 (ROOM_TEMP_DIR) 中的同名文件按普通方式处理（不走低温截断逻辑）。
LOW_TEMP_FALLBACK_USE_ROOM = True

# 并行处理worker上限
MAX_WORKERS = 16

# 绘图 / 平滑相关
PLOT_DPI = 150
PLOT_FIGSIZE: Tuple[float, float] = (6, 4)
SMOOTH_WINDOW = 5
RESIDUAL_MULTIPLIER = 3.0
MIN_DATA_POINTS_FOR_FILTER = 3

# ---- estimate_volume.py 默认设置 ----
MIN_AREA = 30.0
MIN_CIRCULARITY = 0.90

# 体积计算方法选择: 'ellipsoid', 'legendre'
VOLUME_METHOD = 'ellipsoid'


# ---- 低温缺损拟合方式选择 ----
# 当 get_temp.py / estimate_volume.py 在 low_temp_read 分支下处理截断缺损时，
# 可选择使用 'ellipse' (截断椭圆算法) 或 'legendre' (勒让德缺损拟合)。
# 目前 'legendre' 方式会调用占位函数，返回占位结果，便于后续实现。
LOW_TEMP_DEFECT_METHOD = 'ellipse'  # 可选: 'ellipse', 'legendre'



# 勒让德多项式相关参数
LEGENDRE_DEGREE = 6
LEGENDRE_ANGLE_SAMPLES = 180
LEGENDRE_REGULARIZATION = 1e-6

# 包络圆形度检查阈值（包络区域与圆形的最小相似度）
ENVELOPE_MIN_CIRCULARITY = 0.7






# ---- 绘图平滑开关 ----
ENABLE_SMOOTH_FILTERING = False

# ---- Y轴坐标范围控制 ----
# 如果设置为 None，则自动确定范围；如果设置为 (min, max)，则使用固定范围
Y_AXIS_RANGE = (7.5, 9)
if ROOM_TEMP_TIME_MODE:
    Y_AXIS_RANGE = (8, 9.5)

# ---- X轴坐标范围控制 ----
# 如果设置为 None，则自动确定范围；如果设置为 (min, max)，则使用固定范围
# 温度模式下单位为°C，时间模式下单位为分钟
X_AXIS_RANGE = (0, 1300)  # 例如：(800, 1200) 表示温度范围，或 (0, 60) 表示时间范围

if ROOM_TEMP_TIME_MODE:
    X_AXIS_RANGE = None

# ---- 拟合范围控制 ----
# 单次拟合模式下的拟合范围控制（向后兼容）
# 如果设置为 None，则使用所有数据点进行拟合；如果设置为 (min_temp, max_temp)，则只使用该温度范围内的点进行拟合
# 仅在温度模式下生效，时间模式下可使用 TIME_FIT_RANGE
# 示例：
# TEMP_FIT_RANGE = (800, 1100)  # 只使用800°C到1100°C范围内的点进行拟合
# TEMP_FIT_RANGE = (900, 1000)  # 只拟合高温段
# TEMP_FIT_RANGE = None          # 使用所有数据点（默认）
TEMP_FIT_RANGE = None



# ---- 拟合功能控制 ----
# 启用拟合功能：控制是否在图上显示拟合曲线
ENABLE_FITTING = True

# ---- 双重拟合功能控制 ----
# 启用双重拟合：允许在同一图上显示两条不同的拟合曲线（仅在ENABLE_FITTING为True时生效）
ENABLE_DUAL_FIT = True

# 拟合阶数控制
# 对于勒让德多项式拟合，阶数越高拟合越精确但可能出现过拟合
FIT_DEGREE_1 = 1  # 第一次拟合的阶数
FIT_DEGREE_2 = 1  # 第二次拟合的阶数（仅在双重拟合模式下生效）

# 第一次拟合的数据点范围控制
# 温度模式下单位为°C
TEMP_FIT_RANGE_1 = (1085,1300)  # 第一次拟合的温度范围

# 第二次拟合的数据点范围控制（仅在双重拟合模式下生效）
TEMP_FIT_RANGE_2 = (801, 1074)   # 第二次拟合的温度范围

# 第一次拟合曲线的显示范围控制
# 如果设置为 None，则使用数据点范围；如果设置为 (min, max)，则在该范围内显示拟合曲线
TEMP_FIT_DISPLAY_RANGE_1 = (1000, 1300) # 第一次拟合曲线的温度显示范围

# 第二次拟合曲线的显示范围控制（仅在双重拟合模式下生效）
TEMP_FIT_DISPLAY_RANGE_2 = (0,1100)   # 第二次拟合曲线的温度显示范围


# ---- 勒让德多项式拟合相关参数 ----
# 勒让德多项式拟合阶数
LEGENDRE_DEGREE = 6

# 勒让德拟合的角度采样数
LEGENDRE_ANGLE_SAMPLES = 360

# 勒让德拟合的正则化参数（防止过拟合）
LEGENDRE_REGULARIZATION = 1e-6

# 勒让德拟合边界检查：超出边界点比例阈值
# 如果超过此比例的拟合点超出图像边界，则舍弃该拟合
LEGENDRE_BOUNDARY_THRESHOLD = 0.05  # 5%

# 勒让德拟合边界检查：边界容差（像素）
LEGENDRE_BOUNDARY_MARGIN = 5

# 勒让德多项式拟合的数值积分节点数
LEGENDRE_QUAD_NODES = 64

# 勒让德截断检测差值阈值（像素）：拟合半径 - 观测半径 > 该值视为潜在缺损
# 对于缺球冠检测，使用更敏感的阈值
LEGENDRE_TRUNC_DIFF_THRESH_PX = 1.0


LEGENDRE_OVERLAY_MODE = 'both'

# ---- 傅里叶级数拟合相关参数 ----
# 傅里叶拟合的最大谐波数（阶数）
FOURIER_MAX_HARMONICS = 40

# 傅里叶拟合的角度采样数
FOURIER_ANGLE_SAMPLES = 360

# 傅里叶拟合体积计算的数值积分点数
FOURIER_QUAD_POINTS = 1000

# ---- 圆形检测（质量控制）相关参数 ----
# 启用圆形检测功能：检测到不符合要求的圆形时直接舍弃
ENABLE_CIRCLE_FIX = True

# 凸性比阈值：轮廓面积/凸包面积，低于此值认为有明显凹陷，舍弃该图像
FIX_MIN_CONVEXITY = 0.98

# 圆形度比阈值：轮廓面积/外接圆面积，低于此值认为不够圆，舍弃该图像
FIX_MIN_CIRCULARITY = 0.94

# ---- 缺口填充相关参数 ----
# 高光缺口修复的凸性阈值：轮廓面积/凸包面积，低于此值使用凸包填充
SPEC_FIX_MIN_CONVEXITY = 0.80  # 降低阈值，更容易触发凸包填充

# 角度缺口阈值（度）：检测到的最大角度缺口超过此值时使用凸包填充
SPEC_GAP_DEG_THR = 3.0  # 降低阈值，更早触发凸包填充

# 边界安全边距（像素）：距离图像边缘此范围内的区域不做修复
SPEC_FIX_BORDER_MARGIN = 50  # 减少边距，扩大修复范围

# 角度采样精度：用于检测角度缺口的采样点数
SPEC_GAP_BINS = 360

# 形态学处理强度系数：控制小缺口填充的强度
MORPH_CLOSE_FACTOR = 0.05  # 增加系数，填充更大的缺口

# ---- 背景去除相关参数 ----
# 最小球体面积（像素）
BG_MIN_SPHERE_AREA = 100

# 最大球体面积占图像比例
BG_MAX_SPHERE_AREA_RATIO = 0.8

# 球体候选的最小圆形度阈值
BG_MIN_CIRCULARITY = 0.3

# 球体候选的最小紧凑度阈值（面积/外接矩形面积）
BG_MIN_COMPACTNESS = 0.2

# 边界排除比例（接近图像边界的区域将被排除）
BG_BORDER_EXCLUSION = 0.05

# ---- 动态阈值边缘扩展相关参数 ----
# 最大扩展步数：向外扩展的最大次数
EDGE_MAX_EXPAND = 4

# 最小扩展步数：确保的最小扩展次数
EDGE_MIN_EXPAND = 0

# 背景采样步数：用于估算背景灰度基准的最后几步
EDGE_BG_SAMPLE_STEPS = 1

# 灰度差异阈值：当环带灰度与背景基准差异超过此值时认为达到边界
EDGE_GRAY_THRESHOLD = 50

# 形态学操作核大小
EDGE_KERNEL_SIZE = 3

# ---- 拟合曲线特定温度点标注功能 ----
# 启用在拟合曲线上标注特定温度点的功能
ENABLE_TEMP_POINT_ANNOTATION = True

# 第一条拟合曲线上要标注的温度点（单位：°C）
# 设置为 None 则不标注；设置为数值则在该温度点标注密度值
TEMP_POINT_1 = 1085  # 例如：在1200°C处标注第一条拟合曲线的密度值

# 第二条拟合曲线上要标注的温度点（单位：°C）
# 仅在启用双重拟合时生效
TEMP_POINT_2 = 25   # 例如：在900°C处标注第二条拟合曲线的密度值

# 标注点的样式设置
ANNOTATION_MARKER_SIZE = 8          # 标注点的大小
ANNOTATION_MARKER_COLOR_1 = 'red'   # 第一条拟合曲线标注点的颜色
ANNOTATION_MARKER_COLOR_2 = 'green' # 第二条拟合曲线标注点的颜色
ANNOTATION_TEXT_OFFSET_X = 20       # 标注文字的X方向偏移（像素）
ANNOTATION_TEXT_OFFSET_Y = 10       # 标注文字的Y方向偏移（像素）
ANNOTATION_TEXT_SIZE = 10           # 标注文字的大小
ANNOTATION_DECIMAL_PLACES = 3       # 标注数值的小数位数

# ---- 弧形缺损椭球体积计算相关参数 ----
# 椭球体积计算方法: 'radial_integral', 'area_scaled', 'defect_corrected', 'mixed'
ELLIPSE_VOLUME_METHOD = 'radial_integral'

# 椭球径向收缩阈值：收缩因子低于此值认为是缺损区域
ELLIPSE_SHRINK_THRESHOLD = 0.9

# 椭球角度采样数：用于计算径向距离的角度采样点数
ELLIPSE_ANGULAR_SAMPLES = 360

# 椭球φ角采样数：用于三维积分的φ角采样点数
ELLIPSE_PHI_SAMPLES = 180

# 椭球深度影响系数：缺损对三维体积的影响权重
ELLIPSE_DEPTH_IMPACT = 0.8

# 椭球径向角度窗口：用于投影计算的角度窗口大小（度）
ELLIPSE_RADIAL_WINDOW_DEG = 1.0

# 椭球深度纵横比：第三轴（深度）与短半轴的比值
ELLIPSOID_DEPTH_ASPECT = 1.0





