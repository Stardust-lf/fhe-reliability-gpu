# -*- coding: utf-8 -*-
import copy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Gill Sans"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 15

def ntt(a, mod, root):
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        wlen = pow(root, (mod - 1) // length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w % mod
                a[i + j] = (u + v) % mod
                a[i + j + length // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        length *= 2
    return a

# ------------------- 参数设置 -------------------
n = 1024
mod = 15728641
root = 3

# 规律性输入
a = [(i % 16) for i in range(n)]

# 原始NTT
A = ntt(copy.deepcopy(a), mod, root)

# 单点噪声
a_noisy = copy.deepcopy(a)
a_noisy[10] = (a_noisy[10] + 1) % mod
A_noisy = ntt(copy.deepcopy(a_noisy), mod, root)

# -*- coding: utf-8 -*-
import copy
import matplotlib.pyplot as plt

# ------------------- 参数和NTT部分保持不变 -------------------
# ... （ntt函数和a, a_noisy, A, A_noisy部分和你原来的一样） ...

# ------------------- 可视化 -------------------
plt.figure(figsize=(7,6), dpi=150)

# 左图：输入序列 + 注入噪声位置标注
plt.subplot(2,1,1)
plt.plot(a[:128], '-', label='Original Input', color="dimgrey", lw=2)
# 标注噪声位置
noisy_index = 10
if noisy_index < 128:  # 确保在绘图范围内
    plt.scatter(noisy_index, a[noisy_index], color='red', s=80, zorder=5, marker='^')
    # plt.text(noisy_index + 2, a[noisy_index], "+1 Error", color='red', fontsize=15)
plt.title("NTT Input (first 128 elements)")
plt.xlabel("Index")
plt.ylabel("Value")
# plt.legend()
plt.grid(True)

# 右图：NTT输出
plt.subplot(2,1,2)
plt.plot(A[:128], '-', label='Original NTT', lw=2, color="blue")
plt.plot(A_noisy[:128], '-', label='Noisy NTT', color='orange')
plt.title("NTT Output (first 128 elements)")
plt.xlabel("Index")
plt.ylabel("Value (mod {})".format(mod))
plt.legend(loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.savefig("ntt_motivation.png")
