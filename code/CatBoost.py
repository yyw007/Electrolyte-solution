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


# 计算平均绝对百分误差 (MAPE)
def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

# 初始参数设置
test_size = 0.2  # 用作测试集的数据比例
model_seed = 100

path = r"Raw data.csv"  # 请将 'your_file.csv' 替换为你的 CSV 文件路径
save_dir = r"picture"  # 设置图片保存目录

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
pbounds = {'iterations': (50, 2000),
           'depth': (2, 10),
           'learning_rate': (0.01, 0.1)}

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
print("R2 =", r2_score(y_test, predict_y))

# 绘制预测结果对比图
# def plot_results(y_test, y_pred, save_path):
#     plt.figure(figsize=(6, 4))
# #
# #     plt.scatter(y_test, y_pred, alpha=0.8, color='royalblue', s=20, label='Predicted', marker='o')  # 预测值
# #     plt.scatter(y_test, y_test, alpha=0.8, color='LawnGreen', s=20, label='Actual', marker='o')  # 实际值
# #     # Plot the perfect prediction line
# #     min_val = min(y_test.min(), y_pred.min())
# #     max_val = max(y_test.max(), y_pred.max())
# #     plt.plot([min_val, max_val], [min_val, max_val], ':', color='black', lw=2, label='Perfect Prediction')
# #
# #     # Add axis labels and title
# #     plt.xlabel('Experimental', fontsize=15)
# #     plt.ylabel('Predicted', fontsize=15)
# #     plt.title(f'Catboost Prediction\nR²: {r2_score(y_test, y_pred):.2f} - MSE: {mean_squared_error(y_test, y_pred):.2f}', fontsize=15)
# #     plt.legend()
# #     plt.grid(False)
# #     plt.tight_layout()
# #     plt.legend(loc='best', fontsize=11)  # 图例字体大小
# #     plt.tick_params(axis='x', labelsize=15)  # X轴刻度字体大小
# #     plt.tick_params(axis='y', labelsize=15)  # Y轴刻度字体大小
# #     # 保存图像
# #     plt.savefig(f"{save_path}/Catboost_prediction.jpeg", format='jpeg', dpi=600)
# #     # Show the plot
# #     plt.show()
#
# # 调用绘图函数
# plot_results(y_test, predict_y, save_dir)

# 特征重要性和SHAP值
print(model.feature_importances_)
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)
plt.figure(dpi=600)
shap.summary_plot(shap_values, x_train, show=False)
plt.savefig(f"{save_dir}/chushap_summary_plot.png", dpi=600)
plt.figure(dpi=600)
shap.summary_plot(shap_values, x_train, plot_type="bar", show=False)
plt.savefig(f"{save_dir}/chushap_summary_bar_plot.png", dpi=600)

