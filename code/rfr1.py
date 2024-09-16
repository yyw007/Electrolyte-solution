import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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
path = 'merge.csv'  # 请替换为你的CSV文件路径

# 读取CSV文件
df = pd.read_csv(path, encoding='utf-8')

# 数据集划分
cols_list = df.columns[:-1]
target = df.columns[-1]

# 数据预处理
min_max_scaler = MinMaxScaler()
for col in cols_list:
    df[col] = min_max_scaler.fit_transform(df[col].values.reshape(-1, 1))
df[target] = min_max_scaler.fit_transform(df[target].values.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(df[cols_list], df[target], test_size=test_size, random_state=model_seed)

# 定义优化函数，使用交叉验证
def rf_cv(n_estimators, max_features, max_depth):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_features=max_features,
        max_depth=int(max_depth),
        random_state=model_seed
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=model_seed)
    scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=cv)
    return scores.mean()

# 贝叶斯优化
pbounds = {
    'n_estimators': (100, 2000),
    'max_features': (0.1, 0.999),
    'max_depth': (1, 20)
}

optimizer = BayesianOptimization(
    f=rf_cv,
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
best_params['max_features'] = min(max(best_params['max_features'], 0.1), 0.999)
print('Best parameters found by Bayesian Optimization are:', best_params)

# 使用最优参数重新拟合模型
model = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    random_state=model_seed
)
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
r2 = r2_score(y_test, predict_y)

# 输出评估指标
print('RMSE: %.14f' % rmse)
print('MSE: %.14f' % mse)
print('MAE: %.14f' % mae)
print('MAPE: %.14f' % mape)
print("R2 =", r2)

# 绘制预测结果和真实值的图表
df1 = pd.DataFrame({'test': y_test, 'prediction': predict_y})

plt.plot(range(len(df1['test'])), df1['test'], color='blue', label='Target')
plt.plot(range(len(df1['prediction'])), df1['prediction'], color='black', label='Prediction')
plt.legend()
plt.show()
