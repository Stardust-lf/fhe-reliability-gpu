import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 20

def plot_breakdown(ax, df, categories, colors):
    bar_width = 0.3
    x = range(len(df))
    bottom_base = [0.0] * len(df)
    bottom_rfhe = [0.0] * len(df)

    # 绘制 Baseline 和 RFHE 柱状堆叠
    for i, cat in enumerate(categories):
        ax.bar([pos - bar_width/1.5 for pos in x],
               df[cat], width=bar_width, bottom=bottom_base,
               color=colors[i], edgecolor="black", alpha=0.6)
        bottom_base = [bottom_base[j] + df[cat][j] for j in range(len(df))]

        ax.bar([pos + bar_width/1.5 for pos in x],
               df[f"{cat}_rfhe"], width=bar_width, bottom=bottom_rfhe,
               color=colors[i], edgecolor="black", alpha=0.6, hatch='//')
        bottom_rfhe = [bottom_rfhe[j] + df[f"{cat}_rfhe"][j] for j in range(len(df))]

    # 在右侧 twin y-axis 上画 RFHE 高度 - 100 的折线
    ax2 = ax.twinx()
    rfhe_overhead = [val - 100 for val in bottom_rfhe]
    line_overhead, = ax2.plot(
        list(x), rfhe_overhead,
        color="#e41a1c", marker="s", linestyle="--",
        linewidth=1.2, markersize=5,
        markerfacecolor='none', markeredgewidth=1.5,
        label="RFHE overhead"
    )
    ax2.set_ylim(0,10)
    if ax== axes[-1]:
        ax2.set_ylabel("RFHE overhead (%)")
    ax2.tick_params(axis='y')

    # X 轴设置
    ax.set_xticks(list(x))
    ax.set_ylim(0, 115)
    ax.set_xticklabels(df["Polydim"], rotation=45, ha='right')
    ax.set_xlabel("Polynominal dimension")
    ax.grid(True, which="major", ls="--", lw=0.5)

    # 返回折线句柄给外部统一 legend
    return line_overhead



# === 子图1（与你上一步一致的数据1）===
data_bfv_mul = {
    "Polydim": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "NTT": [77.385, 78.5895, 78.991, 79.3925, 79.59325, 80.1955, 80.2758, 80.4364, 80.3],
    "BaseConv": [7.046, 7.1642, 7.6036, 7.683, 7.7227, 7.5218, 7.56968, 7.54544, 7.6],
    "Modmul": [10.569, 10.7463, 11.4054, 11.5245, 11.58405, 11.2827, 11.35452, 11.31816, 11.4],
    "Others": [5, 3.5, 2, 1.4, 1.1, 1, 0.8, 0.7, 0.7],
    "NTT_rfhe": [79.95442383, 80.58928272, 80.59550469, 80.72643239, 80.73325749, 81.19931378, 81.17173527, 81.24973825, 81.0449707],
    "BaseConv_rfhe": [7.073523438, 7.178192578, 7.611025391, 7.686751465, 7.724585425, 7.522718188, 7.570142017, 7.545670269, 7.600115967],
    "Modmul_rfhe": [12.05526563, 12.25749844, 13.00928438, 13.14513281, 13.21305703, 12.86932969, 12.95124938, 12.90977625, 13.003125],
    "Others_rfhe": [10, 7, 4, 2.8, 2.2, 2, 1.6, 1.4, 1.4]
}
df1 = pd.DataFrame(data_bfv_mul)


# === 子图2（你给的第二份数据）===
data_ckks_mul = {
    "Polydim": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "NTT": [66.7, 68.89, 69.62, 70.35, 70.715, 71.81, 72.54, 72.905, 73],
    "BaseConv": [12.735, 12.4245, 12.771, 12.7125, 12.68325, 12.2355, 11.997, 11.87775, 11.835],
    "Modmul": [15.565, 15.1855, 15.609, 15.5375, 15.50175, 14.9545, 14.663, 14.51725, 14.465],
    "Others": [5, 3.5, 2, 1.4, 1.1, 1, 0.8, 0.7, 0.7],
    "NTT_rfhe": [68.91464844, 70.64296995, 71.03415625, 71.53200263, 71.72784505, 72.70885177, 73.34959821, 73.64218398, 73.67724609],
    "BaseConv_rfhe": [12.78474609, 12.4487666, 12.78347168, 12.71870728, 12.6863465, 12.23699359, 11.99773224, 11.87811248, 11.83518059],
    "Modmul_rfhe": [17.75382813, 17.32096094, 17.80401563, 17.72246094, 17.68168359, 17.05747656, 16.72498438, 16.55873828, 16.49914063],
    "Others_rfhe": [10, 7, 4, 2.8, 2.2, 2, 1.6, 1.4, 1.4]
}
df2 = pd.DataFrame(data_ckks_mul)

# === 子图3（你刚给的第三份数据）===

data_rot = {
    "Polydim": [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
    "NTT": [70.35, 71.445, 71.81, 72.175, 72.3575, 72.905, 72.978, 73.124, 73],
    "BaseConv": [8.74, 8.878, 8.924, 8.97, 8.993, 9.062, 9.0712, 9.0896, 9.2],
    "Modmul": [15.96, 16.212, 16.296, 16.38, 16.422, 16.548, 16.5648, 16.5984, 16.8],
    "Others": [5, 3.5, 3, 2.5, 2.25, 1.5, 1.4, 1.2, 1],
    "NTT_rfhe": [72.68583984, 73.26298429, 73.26864063, 73.38766581, 73.39387044, 73.81755798, 73.79248661, 73.86339841, 73.67724609],
    "BaseConv_rfhe": [8.774140625, 8.895339844, 8.932714844, 8.974379883, 8.995195557, 9.063106201, 9.071753662, 9.089877393, 9.200140381],
    "Modmul_rfhe": [18.204375, 18.4918125, 18.587625, 18.6834375, 18.73134375, 18.8750625, 18.894225, 18.93255, 19.1625],
    "Others_rfhe": [10, 7, 6, 5, 4.5, 3, 2.8, 2.4, 2]
}


df3 = pd.DataFrame(data_rot)

categories = ["NTT", "BaseConv", "Modmul", "Others"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(1, 3, figsize=(14, 6), dpi=300, sharey=True)

line1 = plot_breakdown(axes[0], df1, categories, colors)
line2 = plot_breakdown(axes[1], df2, categories, colors)
line3 = plot_breakdown(axes[2], df3, categories, colors)

# 统一图例
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[i], edgecolor="black", label=categories[i]) for i in range(len(categories))]
handles.append(plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="black", label="Baseline"))
handles.append(plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="black", hatch='//', label="RFHE"))
handles.append(line1)  # RFHE overhead 折线
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
axes[0].set_ylabel("Relative complexity")

labels = ['(a)', '(b)', '(c)']
for ax, label in zip(axes, labels):
    ax.text(- 0.12, 1.1, label, transform=ax.transAxes,
            fontsize=25, va='top', ha='left')

plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig("eva_5_mult_polydim.jpg")
# plt.show()
# plt.savefig("three_polydim_breakdown.png", dpi=300)
