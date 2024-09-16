import pandas as pd
from gplearn.genetic import SymbolicRegressor, _Function
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import pearsonr

# 自定义幂函数
def _power(x):
    return np.power(x, 2)

# 创建幂函数
power = _Function(function=_power, name='power', arity=1)

# 读取数据
file_path =  'merge.csv'  # 替换为你本地文件的路径
data = pd.read_csv(file_path)

# 检查并处理异常值和无穷大
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 特征和目标变量
X = data.drop(columns=['Dielectric'])  # 删除最后一列作为特征
y = data['Dielectric']  # 最后一列作为目标变量

# 过滤掉y中包含的无穷大或过大值
y = y[np.isfinite(y) & (np.abs(y) < np.finfo(np.float64).max)]

# 对应去掉X中相应的行
X = X.loc[y.index]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义 function_set，包括高级数学操作和自定义函数
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos', 'tan', power]

# 定义 SymbolicRegressor 模型并设置参数
est_gp = SymbolicRegressor(
    function_set=function_set,
    verbose=1,
    random_state=0,
    stopping_criteria=0.01,  # 设置停止条件
    generations=30,  # 默认值，可以根据需要调整
    population_size=3000,  # 默认值，可以根据需要调整
    parsimony_coefficient=0.03,  # 大幅提高惩罚系数
    max_samples=0.9,  # 样本比例，默认值
    init_depth=(3, 7)  # 初始化树的深度范围
)

# 训练模型
est_gp.fit(X_train, y_train)

# 预测
y_pred = est_gp.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 计算皮尔森相关系数
pearson_corr, _ = pearsonr(y_test, y_pred)
print(f'Pearson Correlation Coefficient: {pearson_corr}')

# 输出最佳表达式
print(f'Best expression: {est_gp._program}')
