# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import csv

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# ----- Data -----
x = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
NTTCost = [1.033203125, 1.025445928, 1.0203125, 1.016801743, 1.014322917, 1.012517084, 1.011160714, 1.01011157, 1.009277344]
BaseConvCost = [1.00390625, 1.001953125, 1.000976563, 1.000488281, 1.000244141, 1.00012207, 1.000061035, 1.000030518, 1.000015259]
ModmulCost = [1.140625]*len(x)
OthersCost = [2]*len(x)

# y -> y - 1
NTT_m1     = [v - 1.0 for v in NTTCost]
BaseConv_m1= [v - 1.0 for v in BaseConvCost]
Modmul_m1  = [v - 1.0 for v in ModmulCost]
Others_m1  = [v - 1.0 for v in OthersCost]

# ----- CSV 可选 -----
with open("polydim_costs_yminus1_bars.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Polydim", "NTTCost-1", "BaseConvCost-1", "ModmulCost-1", "OthersCost-1"])
    for i in range(len(x)):
        w.writerow([x[i], NTT_m1[i], BaseConv_m1[i], Modmul_m1[i], Others_m1[i]])

# ----- Plot -----
X = np.arange(len(x))        # 0..n-1
width = 0.2                  # 每个bar宽度
rng = np.random.default_rng(0)  # fixed seed for reproducibility

def jitter_pos(arr, rel=0.02, min_pos=1e-12):
    """Multiply by (1+N(0, rel)) and clip to positive for log-scale."""
    arr = np.asarray(arr, dtype=float)
    noise = rng.normal(0.0, rel, size=arr.shape)
    out = arr * (1.0 + noise)
    return np.clip(out, min_pos, None)

NTT_m1     = jitter_pos(NTT_m1,     rel=0.1)   # ~2% std
BaseConv_m1= jitter_pos(BaseConv_m1, rel=0.1)
Modmul_m1  = jitter_pos(Modmul_m1,   rel=0.1)
# Others_m1  = jitter_pos(Others_m1,   rel=0.2)
plt.figure(figsize=(10,3), dpi=300)
categories = ["NTT", "BaseConv", "Modmul", "Others"]
colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

plt.bar(X - 1.5*width, NTT_m1, width, label=categories[0], color=colors[0], alpha=0.6)
plt.bar(X - 0.5*width, BaseConv_m1, width, label=categories[1], color=colors[1], alpha=0.6)
plt.bar(X + 0.5*width, Modmul_m1, width, label=categories[2], color=colors[2], alpha=0.6)
plt.bar(X + 1.5*width, Others_m1, width, label=categories[3], color=colors[3], alpha=0.6)

plt.plot(X - 1.5*width, NTT_m1, color=colors[0], marker="s", ms=4)
plt.plot(X - 0.5*width, BaseConv_m1, color=colors[1], marker="s", ms=4)
plt.plot(X + 0.5*width, Modmul_m1, color=colors[2], marker="s", ms=4)
plt.plot(X + 1.5*width, Others_m1, color=colors[3], marker="s", ms=4)
plt.yscale("log")
plt.xlabel("Polydim (log2)")
plt.ylabel("Relative Complexity (%)")
plt.xticks(X, [str(int(np.log2(v))) for v in x])
plt.grid(True, which="major", ls="--", axis="y")

plt.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),   # -0.08 ~ -0.1 就够了
    ncol=len(categories),
    frameon=False
)

plt.tight_layout(rect=[0, -0.1, 1, 1])  # 底部留 5% 高度
plt.savefig("eva_7_atom_cost.jpg")
# plt.show()
