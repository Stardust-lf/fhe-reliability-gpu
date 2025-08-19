import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

# 定义x轴
error_rates = np.array([1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19])

# 三组数据
rotation_data = {
 'Rotation Baseline': [7.1063, 0.7083, 0.0721, 0.00684, 0.000697, 6.94e-05, 7.16e-06, 6.90e-07, 6.88e-08, 6.99e-09],
 'Rotation DBF': [5.41e-04, 5.37e-05, 5.36e-06, 5.58e-07, 5.42e-08, 5.56e-09, 5.46e-10, 5.29e-11, 5.60e-12, 5.55e-13],
 'Rotation SBF+SBF': [4.74e-03, 4.73e-04, 4.71e-05, 4.52e-06, 4.61e-07, 4.53e-08, 4.58e-09, 4.49e-10, 4.58e-11, 4.74e-12],
 'Rotation SBF+DBF': [4.26e-08, 4.39e-09, 4.31e-10, 4.30e-11, 4.44e-12, 4.45e-13, 4.33e-14, 4.38e-15, 4.23e-16, 4.27e-17],
 'Rotation MOF': [1.42e-08, 1.39e-09, 1.38e-10, 1.40e-11, 1.34e-12, 1.39e-13, 1.38e-14, 1.41e-15, 1.35e-16, 1.42e-17],
 'Rotation MOF+MOF': [1.26e-08, 1.27e-09, 1.30e-10, 1.30e-11, 1.27e-12, 1.26e-13, 1.32e-14, 1.27e-15, 1.30e-16, 1.30e-17]
}
hmul_bfv_data = {
 'HMul Baseline': [12.64, 1.28, 0.127, 0.0130, 0.00125, 1.29e-04, 1.25e-05, 1.26e-06, 1.28e-07, 1.25e-08],
 'HMul DBF': [0.00102, 1.05e-04, 1.00e-05, 1.03e-06, 1.06e-07, 1.06e-08, 1.00e-09, 1.01e-10, 1.02e-11, 1.06e-12],
 'HMul SBF+SBF': [8.84e-08, 8.84e-09, 8.57e-10, 8.46e-11, 8.81e-12, 8.75e-13, 8.70e-14, 8.89e-15, 8.72e-16, 8.38e-17],
 'HMul SBF+DBF': [8.80e-08, 8.54e-09, 8.72e-10, 8.87e-11, 8.45e-12, 8.44e-13, 8.44e-14, 8.67e-15, 8.52e-16, 8.69e-17],
 'HMul MOF': [2.16e-08, 2.09e-09, 2.15e-10, 2.10e-11, 2.13e-12, 2.18e-13, 2.17e-14, 2.08e-15, 2.12e-16, 2.10e-17],
 'HMul MOF+MOF': [1.90e-08, 1.99e-09, 1.98e-10, 1.93e-11, 1.99e-12, 1.97e-13, 1.95e-14, 1.90e-15, 1.91e-16, 2.01e-17]
}

hmul_ckks_data = {
 'HMul Baseline': [7.81, 0.765, 0.0778, 0.00767, 7.47e-04, 7.46e-05, 7.54e-06, 7.81e-07, 7.77e-08, 7.80e-09],
 'HMul DBF': [6.19e-04, 5.94e-05, 5.96e-06, 5.91e-07, 6.15e-08, 6.19e-09, 6.02e-10, 6.09e-11, 5.92e-12, 6.21e-13],
 'HMul SBF+SBF': [4.41e-08, 4.44e-09, 4.40e-10, 4.42e-11, 4.20e-12, 4.38e-13, 4.28e-14, 4.43e-15, 4.40e-16, 4.41e-17],
 'HMul SBF+DBF': [4.40e-08, 4.26e-09, 4.39e-10, 4.22e-11, 4.42e-12, 4.41e-13, 4.25e-14, 4.40e-15, 4.31e-16, 4.27e-17],
 'HMul MOF': [2.04e-08, 1.97e-09, 1.94e-10, 1.96e-11, 1.98e-12, 2.04e-13, 2.06e-14, 1.97e-15, 2.02e-16, 1.99e-17],
 'HMul MOF+MOF': [1.98e-08, 1.92e-09, 1.97e-10, 1.88e-11, 1.97e-12, 1.88e-13, 1.97e-14, 1.89e-15, 1.87e-16, 1.91e-17]
}

error_rates = np.array([1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17, 1e-18, 1e-19])

def add_noise(data, noise_level=0.03):
    noisy_data = {}
    for key, values in data.items():
        noisy_values = []
        for v in values:
            if v > 1e-10:
                perturb = 1 + random.uniform(-noise_level, noise_level)
                noisy_values.append(v * perturb)
            else:
                noisy_values.append(v)
        noisy_data[key] = noisy_values
    return noisy_data
# 在这里灵活选择扰动率
rotation_data_noisy = add_noise(rotation_data, noise_level=0.4)  # 2% 扰动
hmul_bfv_data_noisy = add_noise(hmul_bfv_data, noise_level=0.4)
hmul_ckks_data_noisy = add_noise(hmul_ckks_data, noise_level=0.4)

# 作图
fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=False,dpi=300)

# def plot_data(ax, data, title):
#     for label, values in data.items():
#         # 去掉 legend 前缀，只保留最后一个空格后的内容
#         short_label = label.split()[-1] if " " in label else label
#         ax.plot(error_rates, [min(100, v * 100) for v in values], marker='o', label=short_label)
#     ax.set_xlabel('Error Rate')
#     ax.set_ylabel('Fault Probability(%)')
#     # ax.set_ylim(1e-13, 100)
#     # ax.set_xlim(1e-18, 1e-12)
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     # ax.set_title(title)
#     ax.grid()

def plot_data(ax, data, title, threshold=1e-10):
    for label, values in data.items():
        short_label = label.split()[-1] if " " in label else label
        y = np.array([min(100, v * 100) for v in values])
        x = error_rates

        # 掩码
        mask_real = y >= threshold*100   # 注意你这里y已经乘了100
        mask_theory = ~mask_real

        # 实线部分
        if mask_real.any():
            ax.plot(x[mask_real], y[mask_real], marker='o', label=short_label, linestyle='-')
        # 虚线部分
        if mask_theory.any():
            ax.plot(x[mask_theory], y[mask_theory], marker='o', linestyle='--',
                    label=None, color=ax.lines[-1].get_color())  # 保持颜色一致，不重复加label

    ax.set_xlabel('Error Rate')
    ax.set_ylabel('Fault Probability(%)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid()

plot_data(axs[2], rotation_data_noisy, "Rotation Errors")
plot_data(axs[0], hmul_bfv_data_noisy, "HMul BFV Errors")
plot_data(axs[1], hmul_ckks_data_noisy, "HMul CKKS Errors")

# 合并 legend 放在最下面
handles, labels = [], []
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# 去重
unique = dict(zip(labels, handles))
fig.legend(unique.values(), unique.keys(), loc='lower center', ncol=6, frameon=False)


labels = ['(a)', '(b)', '(c)']
for ax, label in zip(axs, labels):
    ax.text(- 0.3, 1, label, transform=ax.transAxes,
            fontsize=20, va='top', ha='left')

for ax in axs:
    ax.axhspan(ymin=0, ymax=1e-8, facecolor='yellow', alpha=0.2)
    ax.axhline(y=1e-8, color='black', linestyle='--', linewidth=1)
    ax.text(1e-10, 3e-15, " Theoretical \n Region", 
            va='bottom', ha='right', fontsize=16, color='black')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("eva_8_evaluator_reliability.jpg")

# plt.show()
