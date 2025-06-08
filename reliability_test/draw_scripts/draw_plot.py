import matplotlib.pyplot as plt
import numpy as np
import re

# 1. 定义 n 列表（从 2 开始，每次翻倍到 2048）
n_values = [2**i for i in range(1, 12)]  # [2, 4, 8, ..., 2048]

# 2. 读取 exp_log.txt，提取所有 "Affected symbols = X/Y (symbol error rate = Z)"
rates = []
pattern = re.compile(r"Affected symbols = \d+/\d+ \(symbol error rate = ([0-9.]+)\)")
with open("symbolerror.txt", "r") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            rates.append(float(m.group(1)))

# 3. 将 rates 重塑为 (len(n_values), 50) 的 numpy 数组
arr = np.array(rates)
arr = arr.reshape(len(n_values), 50)

# 4. 计算每个 n 对应的平均值和标准差
means = arr.mean(axis=1)
stds = arr.std(axis=1, ddof=1)  # ddof=1 计算样本标准差

# 5. 绘制柱状图并加上误差棒
x = np.arange(len(n_values))
plt.figure(figsize=(10, 6))
plt.bar(x, means, yerr=stds, capsize=5)
plt.xticks(x, n_values, rotation=45)
plt.xlabel("n (bit flips)")
plt.ylabel("Mean Symbol Error Rate")
plt.title("Symbol Error Rate vs n (with Std Dev Error Bars)")
plt.tight_layout()
plt.show()
