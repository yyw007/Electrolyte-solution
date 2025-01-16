# 导入必要的库
import pandas as pd
from lifelines import WeibullAFTFitter
import matplotlib.pyplot as plt

# 加载更新后的数据集
file_path = r"80event.csv"
data = pd.read_csv(file_path)

# 获取前 6 列特征和 Class 列
columns = [
    "Class",  # 分组变量
    "Temperature",  # 特征 1
    "Concentration",  # 特征 2
    "Viscosity",  # 特征 3
    "Boiling Point",  # 特征 4
    "Freezing Point",  # 特征 5
    "Molecular mass + (u)"  # 特征 6
]
model_data = data[columns + ["Time", "Event"]]

# 确保 Class 列编码为分类变量
model_data["Class"] = model_data["Class"].astype("category")

# 打印数据预览，确保格式正确
print("\n模型数据预览：")
print(model_data.head())

# 初始化 Weibull AFT 模型
aft_model = WeibullAFTFitter()

# 拟合模型，包含 Class 和 6 个特征
aft_model.fit(
    model_data,
    duration_col="Time",
    event_col="Event",
    formula="Class + Temperature + Concentration + Viscosity + `Boiling Point` + `Freezing Point` + `Molecular mass + (u)`"
)

# 输出模型结果
print("\n模型拟合结果：")
aft_model.print_summary()  # 打印模型系数、显著性水平等

# ----------------------------------
# 补充最终累计风险和生存概率到原数据
# ----------------------------------

# 预测累计风险
cumulative_hazard = aft_model.predict_cumulative_hazard(model_data)

# 为每个样本添加最终的累计风险值（最后一个时间点的值）
model_data["Final Cumulative Hazard"] = cumulative_hazard.iloc[-1].values

# 预测生存概率
survival_function = aft_model.predict_survival_function(model_data)

# 为每个样本添加最终的生存概率（最后一个时间点的值）
model_data["Final Survival Probability"] = survival_function.iloc[-1].values

# 保存补充后的数据
output_file = r"model_data_with_final_cumulative_hazard_and_survival_probability.csv"
model_data.to_csv(output_file, index=False)
print(f"\n结果已保存为 '{output_file}'")

# ----------------------------------
# 可视化：生存曲线和累计风险曲线
# ----------------------------------

# 1. 生存曲线
plt.figure(figsize=(10, 6))
for class_group in model_data["Class"].cat.categories:
    subset = model_data[model_data["Class"] == class_group]
    survival_function = aft_model.predict_survival_function(subset).mean(axis=1)
    plt.plot(survival_function, label=f"Class {class_group}")
plt.title("Survival Curve by Class")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.grid()
plt.show()

# 2. 累计风险曲线
plt.figure(figsize=(10, 6))
for class_group in model_data["Class"].cat.categories:
    subset = model_data[model_data["Class"] == class_group]
    cumulative_hazard = aft_model.predict_cumulative_hazard(subset).mean(axis=1)
    plt.plot(cumulative_hazard, label=f"Class {class_group}")
plt.title("Cumulative Hazard Curve by Class")
plt.xlabel("Time")
plt.ylabel("Cumulative Hazard")
plt.legend()
plt.grid()
plt.show()
