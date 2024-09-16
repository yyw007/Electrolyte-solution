import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
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

path = 'merge.csv'  # 请替换为你的CSV文件路径

# 读取CSV文件
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
def catboost_cv(iterations, depth, learning_rate):
    model = CatBoostRegressor(iterations=int(iterations),
                              depth=int(depth),
                              learning_rate=learning_rate,
                              loss_function='RMSE',
                              verbose=0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return -mean_squared_error(y_test, y_pred)

# 贝叶斯优化
pbounds = {'iterations': (500, 2000),
           'depth': (2, 10),
           'learning_rate': (0.03, 0.1)}

optimizer = BayesianOptimization(
    f=catboost_cv,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=25,
)

best_params = optimizer.max['params']
best_params['iterations'] = int(best_params['iterations'])
best_params['depth'] = int(best_params['depth'])
print('Best parameters found by Bayesian Optimization are:', best_params)

# 使用最优参数重新拟合模型
model = CatBoostRegressor(**best_params)
model.fit(x_train, y_train)

predict_y = model.predict(x_test)

# 反归一化
y_test = np.array(min_max_scaler.inverse_transform([y_test]))
predict_y = min_max_scaler.inverse_transform([predict_y])
y_test = y_test[0, :]
predict_y = predict_y[0, :]

# 计算评估指标
rmse = math.sqrt(mean_squared_error(y_test, predict_y))
mse = mean_squared_error(y_test, predict_y)
mae = mean_absolute_error(y_test, predict_y)
mape = get_mape(y_test, predict_y)
accuracy = 1 - mape

# 输出评估指标
print('RMSE: %.15f' % rmse)
print('MSE: %.15f' % mse)
print('MAE: %.15f' % mae)
print('MAPE: %.15f' % mape)
print('accuracy: %.15f' % accuracy)
print("R2 =", r2_score(y_test, predict_y))

# 特征重要性和SHAP值
print(model.feature_importances_)
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)
shap.summary_plot(shap_values, x_train, plot_type="bar")
shap.dependence_plot("Temperature", shap_values, x_train, interaction_index="Viscosity")

# 绘制预测结果和真实值的图表
plt.plot(range(len(y_test)), predict_y, "o", color='blue', label="predict")
plt.plot(range(len(y_test)), y_test, color='black', label="real")
plt.legend()
plt.show()
