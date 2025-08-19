#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paste-and-plot for resilient-check experiments.

Usage:
  - Paste your CSV rows into CSV_TEXT below and run:
      python paste_plot.py
  - Figures saved to ./plots/plot_<strategy>.png
Notes:
  - Comments are in English by request.
"""

import os
import sys
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15


# ======= PASTE HERE =======
CSV_TEXT = r"""
strategy,mode,fold,miss_rate
intra,SBF,2,0
inter,SBF,2,0
intra,DBF,2,0.24325606
inter,DBF,2,0.24325606
intra,SBF+SBF,2,0
inter,SBF+SBF,2,0.25026218
intra,SBF+DBF,2,0
inter,SBF+DBF,2,0.18910645000000001
intra,MOF,2,0.19996201
inter,MOF,2,0.19996201
intra,MOF+MOF,2,0.040028849999999998
inter,MOF+MOF,2,0.20004026
intra,MOF+SBF,2,0
inter,MOF+SBF,2,0.19994023999999999
intra,MOF+DBF,2,0.048644270000000003
inter,MOF+DBF,2,0.20004150000000001
intra,SBF,4,0
inter,SBF,4,0
intra,DBF,4,0.11503394
inter,DBF,4,0.11503394
intra,SBF+SBF,4,0
inter,SBF+SBF,4,0.12525220000000001
intra,SBF+DBF,4,0
inter,SBF+DBF,4,0.046479329999999999
intra,MOF,4,0.058802960000000001
inter,MOF,4,0.058802960000000001
intra,MOF+MOF,4,0.0034561100000000001
inter,MOF+MOF,4,0.058818380000000003
intra,MOF+SBF,4,0
inter,MOF+SBF,4,0.058821310000000002
intra,MOF+DBF,4,0.0067633099999999998
inter,MOF+DBF,4,0.058810889999999998
intra,SBF,6,0
inter,SBF,6,0
intra,DBF,6,0.071899640000000001
inter,DBF,6,0.071899640000000001
intra,SBF+SBF,6,0
inter,SBF+SBF,6,0.083657110000000007
intra,SBF+DBF,6,0
inter,SBF+DBF,6,0.020269599999999999
intra,MOF,6,0.01537576
inter,MOF,6,0.01537576
intra,MOF+MOF,6,0.00024033
inter,MOF+MOF,6,0.01540419
intra,MOF+SBF,6,0
inter,MOF+SBF,6,0.015370959999999999
intra,MOF+DBF,6,0.0011038899999999999
inter,MOF+DBF,6,0.01538949
intra,SBF,8,0
inter,SBF,8,0
intra,DBF,8,0.051110170000000003
inter,DBF,8,0.051110170000000003
intra,SBF+SBF,8,0
inter,SBF+SBF,8,0.063184379999999998
intra,SBF+DBF,8,0
inter,SBF+DBF,8,0.011401669999999999
intra,MOF,8,0.0038941900000000001
inter,MOF,8,0.0038941900000000001
intra,MOF+MOF,8,1.554e-05
inter,MOF+MOF,8,0.0038925399999999999
intra,MOF+SBF,8,0
inter,MOF+SBF,8,0.0038902799999999999
intra,MOF+DBF,8,0.00019843000000000001
inter,MOF+DBF,8,0.0038992300000000001
intra,SBF,10,0
inter,SBF,10,0
intra,DBF,10,0.038172360000000002
inter,DBF,10,0.038172360000000002
intra,SBF+SBF,10,0
inter,SBF+SBF,10,0.050775870000000001
intra,SBF+DBF,10,0
inter,SBF+DBF,10,0.0072524399999999998
intra,MOF,10,0.00097722
inter,MOF,10,0.00097722
intra,MOF+MOF,10,8.1999999999999998e-07
inter,MOF+MOF,10,0.00097413999999999997
intra,MOF+SBF,10,0
inter,MOF+SBF,10,0.00097623000000000002
intra,MOF+DBF,10,3.6319999999999998e-05
inter,MOF+DBF,10,0.00097612000000000003
intra,SBF,12,0
inter,SBF,12,0
intra,DBF,12,0.029121359999999999
inter,DBF,12,0.029121359999999999
intra,SBF+SBF,12,0
inter,SBF+SBF,12,0.041977849999999997
intra,SBF+DBF,12,0
inter,SBF+DBF,12,0.0048215599999999999
intra,MOF,12,0.00024372
inter,MOF,12,0.00024372
intra,MOF+MOF,12,4.0000000000000001e-08
inter,MOF+MOF,12,0.00024402000000000001
intra,MOF+SBF,12,0
inter,MOF+SBF,12,0.00024688999999999999
intra,MOF+DBF,12,7.0999999999999998e-06
inter,MOF+DBF,12,0.00024127999999999999
intra,SBF,14,0
inter,SBF,14,0
intra,DBF,14,0.02416942
inter,DBF,14,0.02416942
intra,SBF+SBF,14,0
inter,SBF+SBF,14,0.036880400000000001
intra,SBF+DBF,14,0
inter,SBF+DBF,14,0.0037187399999999999
intra,MOF,14,6.1279999999999996e-05
inter,MOF,14,6.1279999999999996e-05
intra,MOF+MOF,14,0
inter,MOF+MOF,14,5.9740000000000001e-05
intra,MOF+SBF,14,0
inter,MOF+SBF,14,6.084e-05
intra,MOF+DBF,14,1.3e-06
inter,MOF+DBF,14,6.003e-05
intra,SBF,16,0
inter,SBF,16,0
intra,DBF,16,0.019587549999999999
inter,DBF,16,0.019587549999999999
intra,SBF+SBF,16,0
inter,SBF+SBF,16,0.03250082
intra,SBF+DBF,16,0
inter,SBF+DBF,16,0.0028589900000000001
intra,MOF,16,1.552e-05
inter,MOF,16,1.552e-05
intra,MOF+MOF,16,0
inter,MOF+MOF,16,1.518e-05
intra,MOF+SBF,16,0
inter,MOF+SBF,16,1.5299999999999999e-05
intra,MOF+DBF,16,1.9999999999999999e-07
inter,MOF+DBF,16,1.525e-05
intra,SBF,18,0
inter,SBF,18,0
intra,DBF,18,0.014900989999999999
inter,DBF,18,0.014900989999999999
intra,SBF+SBF,18,0
inter,SBF+SBF,18,0.028105999999999999
intra,SBF+DBF,18,0
inter,SBF+DBF,18,0.0020048399999999999
intra,MOF,18,3.9299999999999996e-06
inter,MOF,18,3.9299999999999996e-06
intra,MOF+MOF,18,0
inter,MOF+MOF,18,3.63e-06
intra,MOF+SBF,18,0
inter,MOF+SBF,18,3.8600000000000003e-06
intra,MOF+DBF,18,5.9999999999999995e-08
inter,MOF+DBF,18,3.7500000000000001e-06
intra,SBF,20,0
inter,SBF,20,0
intra,DBF,20,0.012679899999999999
inter,DBF,20,0.012679899999999999
intra,SBF+SBF,20,0
inter,SBF+SBF,20,0.025915710000000002
intra,SBF+DBF,20,0
inter,SBF+DBF,20,0.00171148
intra,MOF,20,9.2999999999999999e-07
inter,MOF,20,9.2999999999999999e-07
intra,MOF+MOF,20,0
inter,MOF+MOF,20,9.5999999999999991e-07
intra,MOF+SBF,20,0
inter,MOF+SBF,20,8.9999999999999996e-07
intra,MOF+DBF,20,1e-08
inter,MOF+DBF,20,8.6000000000000002e-07
intra,SBF,22,0
inter,SBF,22,0
intra,DBF,22,0.01133733
inter,DBF,22,0.01133733
intra,SBF+SBF,22,0
inter,SBF+SBF,22,0.024482319999999998
intra,SBF+DBF,22,0
inter,SBF+DBF,22,0.00154974
intra,MOF,22,2.2999999999999999e-07
inter,MOF,22,2.2999999999999999e-07
intra,MOF+MOF,22,0
inter,MOF+MOF,22,2.2999999999999999e-07
intra,MOF+SBF,22,0
inter,MOF+SBF,22,2.9999999999999999e-07
intra,MOF+DBF,22,0
inter,MOF+DBF,22,1.9999999999999999e-07
intra,SBF,24,0
inter,SBF,24,0
intra,DBF,24,0.0098719700000000007
inter,DBF,24,0.0098719700000000007
intra,SBF+SBF,24,0
inter,SBF+SBF,24,0.023000449999999999
intra,SBF+DBF,24,0
inter,SBF+DBF,24,0.0013852199999999999
intra,MOF,24,1.1000000000000001e-07
inter,MOF,24,1.1000000000000001e-07
intra,MOF+MOF,24,0
inter,MOF+MOF,24,5.9999999999999995e-08
intra,MOF+SBF,24,0
inter,MOF+SBF,24,4.0000000000000001e-08
intra,MOF+DBF,24,1e-08
inter,MOF+DBF,24,7.0000000000000005e-08
intra,SBF,26,0
inter,SBF,26,0
intra,DBF,26,0.0083709200000000004
inter,DBF,26,0.0083709200000000004
intra,SBF+SBF,26,0
inter,SBF+SBF,26,0.02155294
intra,SBF+DBF,26,0
inter,SBF+DBF,26,0.0012185799999999999
intra,MOF,26,2e-08
inter,MOF,26,2e-08
intra,MOF+MOF,26,0
inter,MOF+MOF,26,2e-08
intra,MOF+SBF,26,0
inter,MOF+SBF,26,2e-08
intra,MOF+DBF,26,0
inter,MOF+DBF,26,2e-08
intra,SBF,28,0
inter,SBF,28,0
intra,DBF,28,0.0067411399999999996
inter,DBF,28,0.0067411399999999996
intra,SBF+SBF,28,0
inter,SBF+SBF,28,0.020072280000000001
intra,SBF+DBF,28,0
inter,SBF+DBF,28,0.00105708
intra,MOF,28,1e-08
inter,MOF,28,1e-08
intra,MOF+MOF,28,0
inter,MOF+MOF,28,1e-08
intra,MOF+SBF,28,0
inter,MOF+SBF,28,0
intra,MOF+DBF,28,0
inter,MOF+DBF,28,0
intra,SBF,30,0
inter,SBF,30,0
intra,DBF,30,0.00520522
inter,DBF,30,0.00520522
intra,SBF+SBF,30,0
inter,SBF+SBF,30,0.018621599999999999
intra,SBF+DBF,30,0
inter,SBF+DBF,30,0.00090025999999999995
intra,MOF,30,0
inter,MOF,30,0
intra,MOF+MOF,30,0
inter,MOF+MOF,30,0
intra,MOF+SBF,30,0
inter,MOF+SBF,30,0
intra,MOF+DBF,30,0
inter,MOF+DBF,30,0
intra,SBF,32,0
inter,SBF,32,0
intra,DBF,32,0.0037042099999999999
inter,DBF,32,0.0037042099999999999
intra,SBF+SBF,32,0
inter,SBF+SBF,32,0.017159649999999999
intra,SBF+DBF,32,0
inter,SBF+DBF,32,0.00074041999999999999
intra,MOF,32,0
inter,MOF,32,0
intra,MOF+MOF,32,0
inter,MOF+MOF,32,0
intra,MOF+SBF,32,0
inter,MOF+SBF,32,0
intra,MOF+DBF,32,0
inter,MOF+DBF,32,0

"""
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

YSCALE = "log"   # "log" or "linear"
EPS = 1e-12      # epsilon to plot zeros on log scale

def load_df_from_text(text: str) -> pd.DataFrame:
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    buf = io.StringIO("\n".join(lines))
    return pd.read_csv(buf)

def autodetect_cols(df: pd.DataFrame):
    xcol = next((c for c in ["fold","k","x"] if c in df.columns), None)
    ycol = next((c for c in ["miss_rate","error_rate","y"] if c in df.columns), None)
    if xcol is None or ycol is None:
        raise SystemExit(f"Cannot detect x/y columns. Columns: {df.columns.tolist()}")
    if "strategy" not in df.columns:
        df["strategy"] = "unknown"
    if "mode" not in df.columns:
        df["mode"] = "default"
    return xcol, ycol

def to_numeric(df, xcol, ycol):
    df[xcol] = pd.to_numeric(df[xcol], errors="coerce")
    df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
    df.dropna(subset=[xcol, ycol], inplace=True)
    return df

def plot_side_by_side(df, xcol, ycol):
    strategies = sorted(df["strategy"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=False, dpi=300)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    handle_by_label = {}

    for ax, strat in zip(axes, strategies):
        sub = df[df["strategy"] == strat]
        for mode, d in sub.groupby("mode"):
            d = d.sort_values(xcol)
            x = d[xcol].to_numpy()
            y = np.clip(d[ycol].to_numpy(), EPS if YSCALE=="log" else -np.inf, None)
            line, = ax.plot(x, [i * 100 for i in y], marker="o", label=mode)
            handle_by_label.setdefault(mode, line)  # 每个mode只保留一个句柄

        ax.set_title(strat)
        ax.set_xlabel("Fold Width(bit)")
        ax.set_yscale(YSCALE)
        ax.set_ylim(1e-6, 99)
        ax.grid()

        ax.axvline(x=24, color="red", linestyle="--", linewidth=1)

        # 在竖线旁边加竖排文字
        ax.text(
            24,                     # x 坐标
            ax.get_ylim()[1]*0.0001,   # y 坐标，取中间
            "ReliaFHE",
            rotation=90,            # 文字竖着
            va="center", ha="right",
            color="red", 
        )
    labels = ['(a)', '(b)']
    for ax, label in zip(axes, labels):
        ax.text(-0.1, 1, label, transform=ax.transAxes,
                fontsize=16, va='top', ha='left')

    axes[0].set_ylabel("Collusion Rate(%)")

    # 全局 legend 放在画布内底部中央
    labels = list(handle_by_label.keys())
    handles = [handle_by_label[l] for l in labels]
    fig.legend(handles, labels,
               loc="lower center",
               bbox_to_anchor=(0.5, 0.0),            # 画布内底边
               bbox_transform=fig.transFigure,
               ncol=4,
           frameon=False)

    # 给底部留空间给 legend
    plt.subplots_adjust(bottom=0.35)

    # 不用 tight_layout；如需用，改为：fig.tight_layout(rect=[0, 0.22, 1, 1])
    # plt.savefig("eva_2_modmul_collusion_rate.jpg", bbox_inches="tight", pad_inches=0.03)
    plt.show()



def main():
    df = load_df_from_text(CSV_TEXT)
    xcol, ycol = autodetect_cols(df)
    df = to_numeric(df, xcol, ycol)
    plot_side_by_side(df, xcol, ycol)

if __name__ == "__main__":
    main()
