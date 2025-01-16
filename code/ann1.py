from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.utils import check_random_state

# 解决图表中汉字无法显示问题
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 计算平均绝对百分误差 (MAPE)
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 初始参数设置
test_size = 0.2  # 用作测试集的数据比例
model_seed = 100

path = 'merge.csv'  # 请替换为你的CSV文件路径

# 读取CSV文件
df = pd.read_csv(path, encoding='utf-8')

# 数据标签
cols_list = df.columns[:-1]
target = df.columns[-1]

# 数据预处理
min_max_scaler = MinMaxScaler()
for i in range(len(cols_list)):
    df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))

# 数据集拆分
x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=test_size, random_state=model_seed)

# 定义优化函数
activation_mapping = ['identity', 'logistic', 'tanh', 'relu']
solver_mapping = ['lbfgs', 'sgd', 'adam']

def mlp_cv(hidden_layer_sizes, activation, solver, alpha):
    activation_str = activation_mapping[int(activation)]
    solver_str = solver_mapping[int(solver)]
    model = MLPRegressor(
        hidden_layer_sizes=(int(hidden_layer_sizes),),
        activation=activation_str,
        solver=solver_str,
        alpha=alpha,
        random_state=model_seed,
        max_iter=2000
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return -mean_squared_error(y_test, y_pred)

# 贝叶斯优化
pbounds = {
    'hidden_layer_sizes': (5, 50),
    'activation': (0, 3.9999),  # 值的范围使其能正确映射到对应的字符串
    'solver': (0, 2.9999),  # 值的范围使其能正确映射到对应的字符串
    'alpha': (0.0001, 0.1)
}

optimizer = BayesianOptimization(
    f=mlp_cv,
    pbounds=pbounds,
    random_state=1,
    verbose=2
)

optimizer.maximize(
    init_points=5,
    n_iter=25,
)

best_params = optimizer.max['params']
best_params['hidden_layer_sizes'] = int(best_params['hidden_layer_sizes'])
best_params['activation'] = activation_mapping[int(best_params['activation'])]
best_params['solver'] = solver_mapping[int(best_params['solver'])]
print('最优参数: ', best_params)

# 使用最优参数重新拟合模型
model = MLPRegressor(
    hidden_layer_sizes=(best_params['hidden_layer_sizes'],),
    activation=best_params['activation'],
    solver=best_params['solver'],
    alpha=best_params['alpha'],
    random_state=model_seed,
    max_iter=2000
)
model.fit(x_train, y_train)

predict_y = model.predict(x_test)

# 反归一化
y_test = np.array(min_max_scaler.inverse_transform([y_test]))
predict_y = min_max_scaler.inverse_transform([predict_y])
y_test = y_test[0, :]
predict_y = predict_y[0, :]

# 计算评估指标
mape = get_mape(y_test, predict_y)
rmse = np.sqrt(mean_squared_error(y_test, predict_y))
mse = mean_squared_error(y_test, predict_y)
mae = mean_absolute_error(y_test, predict_y)

# 输出评估指标
print('RMSE: %.15f' % rmse)
print('MSE: %.15f' % mse)
print('MAE: %.15f' % mae)
print('MAPE: %.15f' % mape)
print("R2 =", r2_score(y_test, predict_y))
