import re
import pandas as pd
import matplotlib.pyplot as plt

# Plot styling
plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

# 1) 读文件，提取百分比数字
values = []
with open("data/dotprod_16bits_50.txt", "r") as f:
    for line in f:
        m = re.search(r"=\s*([\d\.]+)%", line)
        if m:
            values.append(float(m.group(1)))

# 2) 构造 DataFrame，x 从 1 到 len(values)
df = pd.DataFrame({
    "x": list(range(1, len(values) + 1)),
    "bit_error": values
})

# 3) 绘图
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
ax.scatter(df["x"], df["bit_error"], alpha=0.5)
ax.set_title("Error Sensitivity vs Bit Flips per Symbol")
ax.set_xlabel("Bit Flips per Symbol (run index)")
ax.set_ylabel("Percentage Error (%)")
ax.grid(True)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig("figures/flipimpact_scatter.png")
plt.show()
