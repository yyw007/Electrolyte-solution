#coding:utf8
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import mpl
from bayes_opt import BayesianOptimization

# 解决图表中汉字无法显示问题
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 计算平均绝对百分误差 (MAPE)
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 初始参数设置
test_size = 0.2  # 用作测试集的数据比例
model_seed = 100

# 读取CSV文件
path = 'merge.csv'
df = pd.read_csv(path, encoding='utf-8')

# 数据集划分
cols_list = df.columns[:-1]
target = df.columns[-1]

# 随机划分数据集
x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=test_size, random_state=model_seed)

# 定义优化函数
def adaboost_cv(n_estimators, learning_rate):
    model = AdaBoostRegressor(
        base_estimator=DecisionTreeRegressor(max_depth=7),
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        random_state=model_seed
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return -mean_squared_error(y_test, y_pred)

# 贝叶斯优化
pbounds = {
    'n_estimators': (500, 2000),
    'learning_rate': (0.03, 0.1)
}

optimizer = BayesianOptimization(
    f=adaboost_cv,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=25,
)

best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
print('最优参数: ', best_params)

# 使用最优参数重新拟合模型
model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=7),
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    random_state=model_seed
)
model.fit(x_train, y_train)

# 预测
predict_y = model.predict(x_test)

# 计算评估指标
rmse = math.sqrt(mean_squared_error(y_test, predict_y))  # 计算均方根误差
mse = mean_squared_error(y_test, predict_y)  # 计算均方误差
mae = mean_absolute_error(y_test, predict_y)  # 计算平均绝对误差
mape = get_mape(y_test, predict_y)
r2 = r2_score(y_test, predict_y)

print('RMSE: %.15f' % rmse)
print('MSE: %.15f' % mse)
print('MAE: %.15f' % mae)
print('MAPE: %.15f' % mape)
print("R2 =", r2)

# 绘制真实值和预测值的折线图
df1 = pd.DataFrame({'test': y_test, 'prediction': predict_y})
plt.plot(range(len(df1['test'])), df1['test'], color='blue', label='Target')
plt.plot(range(len(df1['prediction'])), df1['prediction'], 'o', color='black', label='Prediction')
plt.legend()
plt.show()
