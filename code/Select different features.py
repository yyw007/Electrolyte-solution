import pandas as pd
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt

# 读取数据
file_path = 'merge.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 检查数据
print(data.head())

# 定义生存时间
data['Duration'] = data.index + 1

# 构建生存分析数据框，包含所有相关协变量
covariates = [
    'Molecular mass - (u)', 'Ionic radius - (pm)', 'Charge number -'
]
df = data[['Duration', 'Dielectric'] + covariates]  # 包含协变量的DataFrame

# 定义事件
event_threshold = data['Dielectric'].mean()  # 使用均值作为阈值
data['Event'] = (data['Dielectric'] > event_threshold).astype(int)
df['Event'] = data['Event']

# 使用 WeibullAFTFitter 拟合模型
aft = WeibullAFTFitter()
aft.fit(df, 'Duration', event_col='Event')
aft.print_summary()

# 预测累积风险函数
cumulative_hazard = aft.predict_cumulative_hazard(df)
data['Cumulative_Hazard'] = cumulative_hazard.iloc[-1].values  # 取每个样本的最终累积风险值

# 将更新后的数据保存到新的文件
output_file_path = 'Survival Analysis 3-6 Descriptors.csv'  # 替换为你希望保存文件的路径
data.to_csv(output_file_path, index=False)

# 绘制基线生存函数
plt.figure(figsize=(10, 6))
aft.plot()
plt.title("Baseline Survival Function")
plt.show()

# 绘制部分效果图
plt.figure(figsize=(10, 6))
aft.plot_partial_effects_on_outcome(covariates='Molecular mass - (u)', values=[data['Molecular mass - (u)'].min(), data['Molecular mass - (u)'].mean(), data['Molecular mass - (u)'].max()])
plt.title("Partial Effects of Molecular mass - (u) on Survival Function")
plt.show()

plt.figure(figsize=(10, 6))
aft.plot_partial_effects_on_outcome(covariates='Ionic radius - (pm)', values=[data['Ionic radius - (pm)'].min(), data['Ionic radius - (pm)'].mean(), data['Ionic radius - (pm)'].max()])
plt.title("Partial Effects of Ionic radius - (pm) on Survival Function")
plt.show()

plt.figure(figsize=(10, 6))
aft.plot_partial_effects_on_outcome(covariates='Charge number -', values=[data['Charge number -'].min(), data['Charge number -'].mean(), data['Charge number -'].max()])
plt.title("Partial Effects of Charge number - on Survival Function")
plt.show()
