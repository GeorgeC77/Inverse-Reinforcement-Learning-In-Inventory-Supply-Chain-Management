import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 14


# 加载数据：假设CSV格式为 beta, type, value
# type列为"Theory" 或 "IRL"
df = pd.read_csv("data/reward_IRL.csv").values * -1 / 100
theo = [31.92, 24.76, 18.58, 16.98, 14.55, 14.04, 14.56, 15.47, 17.26, 18.73, 20.88]

# 获取所有 beta 值
beta_values = np.arange(0, 2.1, 0.2)

# 准备图形
fig, ax = plt.subplots(figsize=(10, 5))

bar_width = 0.3
x = np.arange(len(beta_values))  # X 轴刻度

for i, beta in enumerate(beta_values):
    group = df[:,i]

    # 理论值
    theory_value = theo[i]
    ax.bar(x[i] - bar_width/1.5, theory_value, width=bar_width, color='0.7', label='Expert' if i == 0 else "")

    # IRL 分布
    irl_values = df[:,i]
    ax.bar(x[i] + bar_width/1.5, np.mean(irl_values), width=bar_width, color='0.8', alpha=0.7, label='AIRL' if i == 0 else "")

    # 箱型图（叠加在IRL柱状图上）
    box = ax.boxplot(irl_values, positions=[x[i] + bar_width/1.5], widths=0.15,
                     patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='lightgray', color='gray'),
                     medianprops=dict(color='black'),
                     whiskerprops=dict(color='gray'),
                     capprops=dict(color='gray'))

# 设置图表格式
ax.set_xticks(x)
ax.set_xticklabels([f"{b:.1f}" for b in beta_values])
ax.set_xlabel(r"$\beta$")
ax.set_ylabel("Average cost")
# ax.set_title("Theoretical vs IRL Reward across different β")
ax.legend()

plt.tight_layout()
plt.savefig("fig/eva_inv.png")
plt.show()
