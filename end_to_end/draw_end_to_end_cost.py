import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# # 原始数据
# data3 = {
#     "Model": ["Bootstrapping", "Resnet_ReLU", "Resnet_SiLU", "MLP", "LoLa", "LeNet"],
#     "NTT&INTT": [79.81, 75.05498, 76.15517, 71.3942, 71.5408, 72.3471],
#     "BaseConv": [11.1, 10.0841, 10.4052, 8.9608, 8.9792, 9.0804],
#     "ModMul": [7.15, 10.4078, 8.76005, 16.4606, 16.4944, 16.6803],
#     "Others": [1.94, 4.45312, 4.67958, 3.1844, 2.9856, 1.8922],
#     "NTT&INTT_rfhe": [80.84753, 76.03069474, 77.14518721, 72.3223246, 72.4708304, 73.2876123],
#     "BaseConv_rfhe": [11.1333, 10.1143523, 10.4364156, 8.9876824, 9.0061376, 9.1076412],
#     "ModMul_rfhe": [7.233655, 10.52957126, 8.862542585, 16.65318902, 16.68738448, 16.87545951],
#     "Others_rfhe": [2.2116, 5.0765568, 5.3347212, 3.630216, 3.403584, 2.157108]
# }
data3 = {
    "Model": ["Bootstrapping", "Resnet_ReLU", "Resnet_SiLU", "MLP", "LoLa", "LeNet"],
    "NTT&INTT": [79.81, 76.52098, 77.62117, 71.3942, 71.5408, 72.3471],
    "BaseConv": [11.1, 10.2681, 10.5892, 8.9608, 8.9792, 9.0804],
    "ModMul": [7.15, 10.7458, 9.09805, 16.4606, 16.4944, 16.6803],
    "Others": [1.94, 2.46512, 2.69158, 3.1844, 2.9856, 1.8922],
    "NTT&INTT_rfhe": [80.5504248, 77.23089144, 78.34128828, 72.05654854, 72.20450859, 73.01828892],
    "BaseConv_rfhe": [11.10016937, 10.26825668, 10.58936158, 8.960936731, 8.979337012, 9.080538556],
    "ModMul_rfhe": [8.15546875, 12.25692813, 10.37746328, 18.77537188, 18.813925, 19.02596719],
    "Others_rfhe": [3.88, 4.93024, 5.38316, 6.3688, 5.9712, 3.7844]
}

df3 = pd.DataFrame(data3)

# 调整最后一组顺序为 LeNet → Lola → MLP
new_order = [0, 1, 2, 5, 4, 3]  
df3 = df3.iloc[new_order].reset_index(drop=True)

# 计算均值行
avg_row = {"Model": "Average"}
for col in df3.columns[1:]:
    avg_row[col] = df3[col].mean()
df3 = pd.concat([df3, pd.DataFrame([avg_row])], ignore_index=True)

# 偏移量（最后均值组也要有）
offsets = [0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.5]
x = [i + off for i, off in zip(range(len(df3)), offsets)]

# 分组
g1 = [0]
g2 = [1, 2]
g3 = [3, 4, 5]
g4 = [6]  # 新的均值组

# 配色
categories3 = ["NTT&INTT", "BaseConv", "ModMul", "Others"]
base_colors3 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(10, 4), dpi=300)
bar_width = 0.3

# 背景色（最后一组也加背景）
# pad = 0.6
# spans = [
#     (min([x[i] for i in g1]) - pad, max([x[i] for i in g1]) + pad),
#     (min([x[i] for i in g2]) - pad, max([x[i] for i in g2]) + pad),
#     (min([x[i] for i in g3]) - pad, max([x[i] for i in g3]) + pad),
#     (min([x[i] for i in g4]) - pad, max([x[i] for i in g4]) + pad)
# ]
# alphas = [0.80, 0.80, 0.80, 0.80]
# for (x0, x1), a in zip(spans, alphas):
#     ax.axvspan(x0, x1, color="#fff2b0", alpha=a, zorder=0)

# 分隔线
boundary_xs = [
    (max([x[i] for i in g1]) + min([x[i] for i in g2])) / 2.0,
    (max([x[i] for i in g2]) + min([x[i] for i in g3])) / 2.0,
    (max([x[i] for i in g3]) + min([x[i] for i in g4])) / 2.0
]
for bx in boundary_xs:
    ax.axvline(bx, color="#bfbfbf", linewidth=1.2, linestyle="--", zorder=1)

# 柱状图
bottom_normal = [0] * len(df3)
bottom_rfhe = [0] * len(df3)
for i, cat in enumerate(categories3):
    base_color = base_colors3[i]
    ax.bar([pos - bar_width/1.2 for pos in x], df3[cat],
           width=bar_width, bottom=bottom_normal,
           color=base_color, edgecolor="black", alpha=0.6, zorder=2)
    bottom_normal = [bottom_normal[j] + df3[cat][j] for j in range(len(df3))]
    
    ax.bar([pos + bar_width/2 for pos in x], df3[f"{cat}_rfhe"],
           width=bar_width, bottom=bottom_rfhe,
           color=base_color, edgecolor="black", hatch='//', alpha=0.6, zorder=2)
    bottom_rfhe = [bottom_rfhe[j] + df3[f"{cat}_rfhe"][j] for j in range(len(df3))]

# X 轴标签
ax.set_xticks(x)
ax.set_xticklabels(df3["Model"], rotation=15,)

# 分组标题
trans = blended_transform_factory(ax.transData, ax.transAxes)
ax.text(sum([x[i] for i in g1]) / len(g1), -0.22, " ", ha="center", va="top", transform=trans)
ax.text(sum([x[i] for i in g2]) / len(g2), -0.22, "CIFAR-10 Classification", ha="center", va="top", transform=trans)
ax.text(sum([x[i] for i in g3]) / len(g3), -0.22, "MNIST Classification", ha="center", va="top", transform=trans)
ax.text(sum([x[i] for i in g4]) / len(g4), -0.22, " ", ha="center", va="top", transform=trans)

# 主轴
ax.set_ylabel("Normalized complexity(%)", fontweight='bold')
ax.grid(True, which="major", ls="--", lw=0.5, zorder=1)
ax.set_ylim(0, 115)

# 图例
handles = []
for i, cat in enumerate(categories3):
    base_color = base_colors3[i]
    handles.append(plt.Rectangle((0,0),1,1, facecolor=base_color, alpha=1.0, label=cat, edgecolor='black'))
handles.append(plt.Rectangle((0,0),1,1, facecolor='white', alpha=1, label="Baseline", edgecolor='black'))
handles.append(plt.Rectangle((0,0),1,1, facecolor='white', alpha=1, label="ReliaFHE", edgecolor='black', hatch='//'))
ax.legend(handles=handles, loc='lower right', ncol=3)

# 给所有 RFHE 柱子加标注
for xpos, height in zip(x, bottom_rfhe):
    ax.text(
        xpos + 0.3,               # 右侧 RFHE 柱子
        height + 1,               # 顶部向上 1
        f"{height/100:.4f}×",     # 转成比例
        ha="center", va="bottom",
        fontsize=15, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("eva_3_complexity_end_to_end.jpg")
# plt.show()
