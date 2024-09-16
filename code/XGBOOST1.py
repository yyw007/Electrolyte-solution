#coding:utf8
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from pylab import mpl
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
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

# 数据预处理
min_max_scaler = MinMaxScaler()
for i in range(len(cols_list)):
    df[cols_list[i]] = min_max_scaler.fit_transform(df[cols_list[i]].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=test_size, random_state=model_seed)

# 定义优化函数
def xgb_cv(n_estimators, max_depth, learning_rate):
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        seed=model_seed
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return -mean_squared_error(y_test, y_pred)

# 贝叶斯优化
pbounds = {
    'n_estimators': (500, 2000),
    'max_depth': (2, 10),
    'learning_rate': (0.03, 0.1)
}

optimizer = BayesianOptimization(
    f=xgb_cv,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=25,
)

best_params = optimizer.max['params']
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])
print('最优参数: ', best_params)

# 使用最优参数重新拟合模型
model = XGBRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate'],
    seed=model_seed
)
model.fit(x_train, y_train)

predict_y = model.predict(x_test)

# 反归一化
y_test = np.array(min_max_scaler.inverse_transform([y_test]))
predict_y = min_max_scaler.inverse_transform([predict_y])
y_test = y_test[0, :]
predict_y = predict_y[0, :]

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

# 特征重要性和SHAP值
print(model.feature_importances_)
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)

# 绘制SHAP值图
shap.summary_plot(shap_values, x_train)
shap.summary_plot(shap_values, x_train, plot_type="bar")
# 确保 "t" 和 "mass" 是你的特征中的列名
# 如果不是，请替换为实际存在的列名
if "Temperature" in x_train.columns and "Viscosity" in x_train.columns:
    shap.dependence_plot("Temperature", shap_values, x_train, interaction_index="Viscosity")

# 绘制预测结果和真实值的图表
plt.plot(range(len(y_test)), predict_y, "o", color='blue', label="predict")
plt.plot(range(len(y_test)), y_test, color='black', label="real")
plt.legend()
plt.show()
