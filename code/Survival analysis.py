import pandas as pd
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt

# 读取数据
file_path = r'C:\Users\ikun\Desktop\第二\介电常数csv\合并.csv'  # 替换为你的文件路径
data = pd.read_csv(file_path)

# 将 Reduction in Activity Coefficient- 变量分成三个类别
data['Category'] = pd.qcut(data['Reduction in Activity Coefficient-'], q=3, labels=[1, 2, 3])

# 检查类别分布
category_count = data['Category'].value_counts()
print(f"Category distribution:\n{category_count}")

# 定义生存时间
data['Duration'] = data.index + 1

# 构建生存分析数据框
df = data[['Duration', 'Category', 'Dielectric']]  # 'Dielectric' 作为协变量

# 使用 WeibullAFTFitter 拟合模型
aft = WeibullAFTFitter()
aft.fit(df, 'Duration', event_col='Category')
aft.print_summary()

# 预测累积风险函数
cumulative_hazard = aft.predict_cumulative_hazard(df)
data['Cumulative_Hazard'] = cumulative_hazard.iloc[-1].values  # 取每个样本的最终累积风险值

# 将更新后的数据保存到新的文件
output_file_path = 'Survival analysis.csv'  # 替换为你希望保存文件的路径
data.to_csv(output_file_path, index=False)

# 绘制基线生存函数
plt.figure(figsize=(10, 6))
aft.plot()
plt.title("Baseline Survival Function")
plt.show()

# 绘制预测的累积风险函数
plt.figure(figsize=(10, 6))
aft.plot_partial_effects_on_outcome(covariates='Dielectric', values=[60, 70, 80, 90])
plt.title("Partial Effects of Dielectric on Survival Function")
plt.show()
