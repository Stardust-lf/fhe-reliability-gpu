import pandas as pd
import matplotlib.pyplot as plt

# 全局字体设置
plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# 数据
data = {
    "Polydim": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "NTT": [66.7, 68.89, 69.62, 70.35, 70.715, 71.81, 72.54, 72.905, 73],
    "BaseConv": [8.28, 8.556, 8.648, 8.74, 8.786, 8.924, 9.016, 9.062, 9.2],
    "Modmul": [15.12, 15.624, 15.792, 15.96, 16.044, 16.296, 16.464, 16.548, 16.8],
    "Others": [10, 7, 6, 5, 4.5, 3, 2, 1.5, 1],
    "NTT_rfhe": [68.08958333, 70.16574074, 70.78033333, 71.41590909, 71.69715278, 72.73064103, 73.40357143, 73.71505556, 73.76041667],
    "BaseConv_rfhe": [8.31234375, 8.572710938, 8.656445313, 8.744267578, 8.78814502, 8.925089355, 9.016550293, 9.06227655, 9.200140381],
    "Modmul_rfhe": [15.47472107, 15.9903663, 16.16221536, 16.33410816, 16.4200542, 16.67794916, 16.84988089, 16.93584671, 17.1937515],
    "Others_rfhe": [11.4, 7.98, 6.84, 5.7, 5.13, 3.42, 2.28, 1.71, 1.14]
}
df = pd.DataFrame(data)

# 分类与颜色
categories = ["NTT", "BaseConv", "Modmul", "Others"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
bar_width = 0.3
x = range(len(df))

bottom_normal = [0] * len(df)
bottom_rfhe = [0] * len(df)

# 绘制 breakdown 堆叠柱
for i, cat in enumerate(categories):
    # baseline
    ax.bar([pos - bar_width/1.5 for pos in x],
           df[cat],
           width=bar_width,
           bottom=bottom_normal,
           color=colors[i],
           edgecolor="black",
           alpha=0.6,
           label=cat)
    bottom_normal = [bottom_normal[j] + df[cat][j] for j in range(len(df))]

    # RFHE
    ax.bar([pos + bar_width/1.5 for pos in x],
           df[f"{cat}_rfhe"],
           width=bar_width,
           bottom=bottom_rfhe,
           color=colors[i],
           edgecolor="black",
           alpha=0.6,
           hatch='//')
    bottom_rfhe = [bottom_rfhe[j] + df[f"{cat}_rfhe"][j] for j in range(len(df))]

# 次轴数据（log 轴）
complexity_vals = [813, 1752, 3495, 7288, 14377, 31365, 65419, 136819, 281739]
ax2 = ax.twinx()
ax2.plot([pos + bar_width/1.5 for pos in x], complexity_vals,
         color="purple", marker="s", linewidth=2, label="Complexity (MFLOPS)")
# ax2.set_yscale("log")
ax2.set_ylim(-20000, 300000)
ax2.set_ylabel("Latency($\mu s$)")

# 主轴设置
ax.set_xticks(x)
ax.set_xticklabels(df["Polydim"])
ax.set_xlabel("Polynomial Dimension")
ax.set_ylabel("Normalized computational complexity")
ax.grid(True, which="major", ls="--", lw=0.5)

# 图例
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# 添加 Baseline / RFHE 样式说明
handles1.append(plt.Rectangle((0,0),1,1,facecolor="white",edgecolor="black",label="Baseline"))
handles1.append(plt.Rectangle((0,0),1,1,facecolor="white",edgecolor="black",hatch="//",label="RFHE"))

ax.legend(handles1 + handles2, labels1 + labels2, loc="lower right", ncol=3)

plt.tight_layout()
plt.savefig("eva_4_rotation_polydim.jpg")
# plt.show()
