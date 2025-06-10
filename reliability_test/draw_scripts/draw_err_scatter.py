import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

# 读取CSV文件
df = pd.read_csv("data/flipimpact.csv")

# 拆分为两个子集
df_flip = df[df["type"] == "flip_per_symbol"]
df_symbols = df[df["type"] == "num_symbols"]

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, dpi=150)

# 左图：flip_per_symbol
axes[0].scatter(df_flip["x"], df_flip["bit_error"], alpha=0.5)
axes[0].set_title("Bit Error vs Bit Flips per Symbol")
axes[0].set_xlabel("Bit Flips in One Symbol")
axes[0].set_ylabel("Bit Error Rate")
axes[0].grid(True)

# 右图：num_flipped_symbols
# print(df_symbols["bit_error"])
axes[1].scatter(df_symbols["x"], df_symbols["bit_error"], alpha=0.5, color='orange')
axes[1].set_title("Bit Error vs Number of Flipped Symbols")
axes[1].set_xlabel("Number of Flipped Symbols")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("./figures/bit_error.png")
plt.show()
