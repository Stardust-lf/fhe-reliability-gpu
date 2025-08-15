import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# Barrett（去掉 1elem1flip）
barret_files = [
    ("barret_1element_2flip.log", "Barrett"),
    ("barret_2element_1flip.log", "Barrett"),
]
# Montgomery（去掉 1elem1flip）
mont_files = [
    ("mont_1element_2flip.log", "Montgomery"),
    ("mont_2element_1flip.log", "Montgomery"),
]

def prep_ratios(df: pd.DataFrame, scheme_key: str):
    df = df.copy()
    def scheme(df_cond):
        g = df_cond.sort_values("FOLD_WIDTH")[["FOLD_WIDTH","TP","FP","TN","FN"]].copy()
        g["FNR"] = g["FN"] / (g["FN"] + g["TP"])
        g["FPR"] = g["FP"] / (g["TN"] + g["FP"])
        return g[["FOLD_WIDTH","FNR","FPR"]]
    sc    = scheme(df[df[scheme_key] == 1])
    final = scheme(df[df["USE_FINAL"] == 1])
    both  = sc.merge(final, on="FOLD_WIDTH", suffixes=(f"_{scheme_key}","_FINAL"))
    both["FNR_BOTH"] = both[f"FNR_{scheme_key}"] * both["FNR_FINAL"]
    both["FPR_BOTH"] = both[f"FPR_{scheme_key}"] * both["FPR_FINAL"]
    return sc, final, both

fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, dpi=150)
ax1, ax2 = ax

colors = {
    "USE_SN":    "tab:blue",
    "USE_MP":    "tab:orange",
    "USE_FINAL": "tab:green",
    "USE_BOTH":  "tab:red",
}

# 去重 legend
seen_labels = set()
def plot_once(ax, x, y, label, color, marker="o"):
    lab = None if label in seen_labels else label
    ln, = ax.plot(x, y, marker=marker, color=color, label=lab)
    seen_labels.add(label)
    return ln

# --- Subplot 1: Barrett FNR ---
for path, _ in barret_files:
    df = pd.read_csv(path, skipinitialspace=True)
    sn, final, both = prep_ratios(df, "USE_SN")
    plot_once(ax1, sn["FOLD_WIDTH"],    sn["FNR"],        "USE_SN",    colors["USE_SN"])
    plot_once(ax1, final["FOLD_WIDTH"], final["FNR"],     "USE_FINAL", colors["USE_FINAL"])
    plot_once(ax1, both["FOLD_WIDTH"],  both["FNR_BOTH"], "USE_BOTH",  colors["USE_BOTH"])

ax1.set_xlabel("Fold width(bit)")
ax1.set_ylabel("Barrett Modmul. Error Rate")
ax1.set_yscale("log")
ax1.grid(True)

# --- Subplot 2: Montgomery FNR ---
for path, _ in mont_files:
    df = pd.read_csv(path, skipinitialspace=True)
    mp, final, both = prep_ratios(df, "USE_MP")
    plot_once(ax2, mp["FOLD_WIDTH"],    mp["FNR"],        "USE_MP",    colors["USE_MP"])
    plot_once(ax2, final["FOLD_WIDTH"], final["FNR"],     "USE_FINAL", colors["USE_FINAL"])
    plot_once(ax2, both["FOLD_WIDTH"],  both["FNR_BOTH"], "USE_BOTH",  colors["USE_BOTH"])

ax2.set_xlabel("Fold width(bit)")
ax2.set_ylabel("Montgomery Modmul. Error Rate")
ax2.set_yscale("log")
ax2.grid(True)

# 合并 legend
handles, labels = [], []
for a in [ax1, ax2]:
    h, l = a.get_legend_handles_labels()
    handles.extend(h); labels.extend(l)
fig.legend(handles, labels, loc="lower center", ncol=4)

plt.tight_layout(rect=[0,0.12,1,1])
plt.savefig("eva_2_modmul_collusion_rate")
# plt.show()
