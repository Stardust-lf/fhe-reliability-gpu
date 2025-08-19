# -*- coding: utf-8 -*-
# Purpose: 2x2 subplots. Top row = pbits plots (Left: Stage 3 only | Right: Stage 2 only).
#          Bottom row = W plots   (Left: Stage 3 only | Right: Stage 2 only).
# Notes:
#   - No external files needed. No seaborn.
#   - Values shown in percent; include zero-valued curves via symlog.

import matplotlib.pyplot as plt

# ---- Global style ----
plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# ---- Data: pbits (top row) ----
pbits = list(range(2, 31))

series_sparse_pbits = {
    'SCF-BF-S1': {},
    'SCF-BF-S2': {},
    'SCF-BF-S3': {},
    'SCF-MBU-S1': {2: 0.334763, 4: 0.025717},
    'SCF-MBU-S2': {},
    'SCF-MBU-S3': {2: 0.338775, 4: 0.025632},
    'MCF-PPE-S1': {},
    'MCF-PPE-S2': {},
    'MCF-PPE-S3': {},
    'MCF-CTE-S1': {},
    'MCF-CTE-S2': {},
    'MCF-CTE-S3': {},
    'MCF-CLE-S1': {
        2: 0.553784, 3: 0.264624, 4: 0.147322, 5: 0.068000, 6: 0.037321,
        7: 0.017865, 8: 0.012173, 9: 0.004429, 10: 0.002422, 11: 0.001846,
        12: 0.000845, 13: 0.000333, 14: 0.000234, 15: 0.000129, 16: 0.000063,
        17: 0.000018, 18: 0.000012, 19: 0.000003
    },
    'MCF-CLE-S2': {
        2: 0.332733, 3: 0.143369, 4: 0.076443, 5: 0.034525, 6: 0.018587,
        7: 0.008821, 8: 0.005983, 9: 0.002262, 10: 0.001182, 11: 0.000987,
        12: 0.000396, 13: 0.000150, 14: 0.000171, 15: 0.000075, 16: 0.000024,
        17: 0.000021, 18: 0.000012, 19: 0.000015, 20: 0.000012, 21: 0.000009,
        22: 0.000018, 23: 0.000009, 24: 0.000015, 25: 0.000006, 26: 0.000015,
        27: 0.000012, 28: 0.000003, 29: 0.000021
    },
    'MCF-CLE-S3': {
        2: 0.549910, 3: 0.265629, 4: 0.148627, 5: 0.068553, 6: 0.038369,
        7: 0.017665, 8: 0.012020, 9: 0.004585, 10: 0.002408, 11: 0.001751,
        12: 0.000691, 13: 0.000222, 14: 0.000165, 15: 0.000090, 16: 0.000063,
        17: 0.000030, 18: 0.000027, 19: 0.000003, 20: 0.000006, 21: 0.000003,
        22: 0.000006, 23: 0.000003
    },
}

series_pbits = {
    lbl: [series_sparse_pbits[lbl].get(p, 0.0) for p in pbits]
    for lbl in series_sparse_pbits
}

# ---- Data: W (bottom row, pbits fixed at 30) ----
W_values = list(range(2, 25))  # W=2..24

series_sparse_W = {
    'SCF-BF-S1': {},
    'SCF-BF-S2': {},
    'SCF-BF-S3': {},

    'SCF-MBU-S1': {},
    'SCF-MBU-S2': {2: 0.554117, 3: 0.134553},
    'SCF-MBU-S3': {},

    'MCF-PPE-S1': {},
    'MCF-PPE-S2': {},
    'MCF-PPE-S3': {},

    'MCF-CTE-S1': {},
    'MCF-CTE-S2': {},
    'MCF-CTE-S3': {},

    'MCF-CLE-S1': {},
    'MCF-CLE-S2': {
        2: 0.554117, 3: 0.509889, 4: 0.200001, 5: 0.063154, 6: 0.131488,
        7: 0.342829, 8: 0.022611, 9: 0.337504, 10: 0.006250, 11: 0.001935,
        12: 0.013808, 13: 0.000258, 14: 0.008704, 15: 0.000375, 16: 0.000162,
        17: 0.000036, 18: 0.003849, 19: 0.000003, 20: 0.000024, 21: 0.332511,
        22: 0.000003
    },
    'MCF-CLE-S3': {2: 0.009002, 3: 0.000015},
}

series_W = {
    lbl: [series_sparse_W[lbl].get(w, 0.0) for w in W_values]
    for lbl in series_sparse_W
}

# ---- Helper ----
def _plot_group(ax, x_vals, series_dict, stages=('S3',)):
    """Plot selected stages; zeros included; y in percent."""
    for label in sorted(series_dict.keys()):
        if any(label.endswith(f'-{s}') for s in stages):
            clean_label = label.rsplit('-', 1)[0]  # 去掉最后的 -S2 / -S3
            y = series_dict[label]
            ax.plot(x_vals, [100.0 * t for t in y], marker='o',
                    linewidth=1.5, markersize=4, label=clean_label)
    ax.set_yscale('symlog', linthresh=1e-4)  # percent units
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    # 不在这里加 legend

# ---- Plot: 2x2 ----
fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=300, sharey=False, sharex=True)

# Top-left: Stage 3 vs pbits
_plot_group(axes[0, 0], pbits, series_pbits, stages=('S3',))
axes[0, 0].set_ylabel('Collision Probability (%)')

# Top-right: Stage 2 vs pbits
_plot_group(axes[1, 0], pbits, series_pbits, stages=('S2',))
axes[1, 0].set_xlabel('Prime bits')
axes[1, 0].set_ylabel('Collision Probability (%)')

# Bottom-left: Stage 3 vs W
_plot_group(axes[0, 1], W_values, series_W, stages=('S3',))
axes[0, 1].set_ylabel('Collision Probability (%)')
axes[0, 1].set_xlim(0, 24)

# Bottom-right: Stage 2 vs W
_plot_group(axes[1, 1], W_values, series_W, stages=('S2',))
axes[1, 1].set_ylabel('Collision Probability (%)')
axes[1, 1].set_xlabel('Fold bits')

# ---- Legend only on top-right ----
handles, labels = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# 去掉重复的 legend
uniq = dict(zip(labels, handles))
axes[0, 1].legend(uniq.values(), uniq.keys(), loc="upper right", ncol=1)

labels = ['(a)', '(b)', '(c)', '(d)']
for ax, label in zip(axes.flat, labels):
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
             va='top', ha='right', fontweight='bold', fontsize=20)

fig.tight_layout()
plt.savefig("eva_8_ntt_reliability.jpg")
# plt.show()
