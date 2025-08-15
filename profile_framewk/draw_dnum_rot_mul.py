import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

def plot_breakdown(ax, df, categories, colors, xkey, xlabel):
    bar_width = 0.3
    x = range(len(df))
    bottom_base = [0.0] * len(df)
    bottom_rfhe = [0.0] * len(df)

    for i, cat in enumerate(categories):
        ax.bar([pos - bar_width/1.5 for pos in x],
               df[cat], width=bar_width, bottom=bottom_base,
               color=colors[i], edgecolor="black", alpha=0.6)
        bottom_base = [bottom_base[j] + df[cat][j] for j in range(len(df))]

        ax.bar([pos + bar_width/1.5 for pos in x],
               df[f"{cat}_rfhe"], width=bar_width, bottom=bottom_rfhe,
               color=colors[i], edgecolor="black", alpha=0.6, hatch='//')
        bottom_rfhe = [bottom_rfhe[j] + df[f"{cat}_rfhe"][j] for j in range(len(df))]

    # twin y-axis: RFHE overhead（总高度-100）
    ax2 = ax.twinx()
    rfhe_overhead = [val - 100 for val in bottom_rfhe]
    line_overhead, = ax2.plot(
        list(x), rfhe_overhead,
        color="#e41a1c", marker="s", linestyle="--",
        linewidth=1.2, markersize=5,
        markerfacecolor='none', markeredgewidth=1.5,
        label="RFHE overhead"
    )
    ax2.set_ylabel("RFHE overhead (%)")
    ax2.tick_params(axis='y')
    ax2.set_ylim(0, 6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df[xkey], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylim(0, 115)
    ax.grid(True, which="major", ls="--", lw=0.5)

    return line_overhead

# (a)
data_bfv_mul = {
    "dnum": [1, 2, 3, 4, 6, 8, 12, 24],
    "NTT": [40, 50, 60, 65, 71, 75, 78, 80.3],
    "BaseConv": [35, 30, 26, 21, 16, 13, 10, 7.6],
    "Modmul": [23.9, 18.9, 12.9, 12.9, 11.9, 10.9, 10.9, 11],
    "Others": [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    "NTT_rfhe": [40.37109375, 50.46386719, 60.55664063, 65.60302734, 71.65869141, 75.69580078, 78.72363281, 81.0449707],
    "BaseConv_rfhe": [35.00053406, 30.00045776, 26.00039673, 21.00032043, 16.00024414, 13.00019836, 10.00015259, 7.600115967],
    "Modmul_rfhe": [27.2609375, 21.5578125, 14.7140625, 14.7140625, 13.5734375, 12.4328125, 12.4328125, 12.546875],
    "Others_rfhe": [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]
}
df_a = pd.DataFrame(data_bfv_mul)

# (b)
data_ckks_mul = {
    "dnum": [1, 2, 3, 4, 6, 8, 12, 24],
    "NTT": [36.6, 42.6, 48.2, 58.6, 62.6, 69.2, 72.1, 73],
    "BaseConv": [52, 45, 39, 28.2, 23, 15, 11, 8.835],
    "Modmul": [10.3, 11.3, 11.7, 12.1, 13.3, 14.7, 15.8, 17.065],
    "Others": [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    "NTT_rfhe": [36.93955078, 42.99521484, 48.64716797, 59.14365234, 63.18076172, 69.84199219, 72.76889648, 73.67724609],
    "BaseConv_rfhe": [52.00079346, 45.00068665, 39.00059509, 28.2004303, 23.00035095, 15.00022888, 11.00016785, 8.835134811],
    "Modmul_rfhe": [11.7484375, 12.8890625, 13.3453125, 13.8015625, 15.1703125, 16.7671875, 18.021875, 19.46476563],
    "Others_rfhe": [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]
}

df_b = pd.DataFrame(data_ckks_mul)


# (c) 你刚提供的数据
data_rot = {
    "dnum": [1, 2, 3, 4, 6, 8, 12, 24],
    "NTT": [36.6, 42.6, 48.2, 58.6, 62.6, 69.2, 72.1, 73],
    "BaseConv": [55, 48, 42, 31.2, 26, 18, 14, 11.835],
    "Modmul": [7.3, 8.3, 8.7, 9.1, 10.3, 11.7, 12.8, 14.065],
    "Others": [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1],
    "NTT_rfhe": [36.93955078, 42.99521484, 48.64716797, 59.14365234, 63.18076172, 69.84199219, 72.76889648, 73.67724609],
    "BaseConv_rfhe": [55.00083923, 48.00073242, 42.00064087, 31.20047607, 26.00039673, 18.00027466, 14.00021362, 11.83518059],
    "Modmul_rfhe": [8.3265625, 9.4671875, 9.9234375, 10.3796875, 11.7484375, 13.3453125, 14.6, 16.04289063],
    "Others_rfhe": [2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2, 2.2]
}
df_c = pd.DataFrame(data_rot)

categories = ["NTT", "BaseConv", "Modmul", "Others"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=300, sharey=True)

line1 = plot_breakdown(axes[0], df_a, categories, colors, "dnum", "Dnum")
line2 = plot_breakdown(axes[1], df_b, categories, colors, "dnum", "Dnum")
line3 = plot_breakdown(axes[2], df_c, categories, colors, "dnum", "Dnum")

axes[0].set_ylabel("Relative complexity")

labels = ['(a)', '(b)', '(c)']
for ax, label in zip(axes, labels):
    ax.text(-0.1, 1, label, transform=ax.transAxes,
            fontsize=16, va='top', ha='left')

# 统一图例
handles = [plt.Rectangle((0,0),1,1, facecolor=colors[i], edgecolor="black", label=categories[i]) for i in range(len(categories))]
handles.append(plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="black", label="Baseline"))
handles.append(plt.Rectangle((0,0),1,1, facecolor="white", edgecolor="black", hatch='//', label="RFHE"))
handles.append(line1)  # RFHE overhead 折线
fig.legend(handles=handles, loc="lower center", ncol=8)

plt.tight_layout(rect=[0, 0.08, 1, 1])
# plt.show()
plt.savefig("eva_6_mult_dnum.jpg")
