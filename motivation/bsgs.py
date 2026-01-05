import numpy as np
import matplotlib.pyplot as plt

# ------------------- 简单模 NTT/INTT -------------------
def ntt(a, mod, root):
    n = len(a)
    j = 0
    a = a.copy()
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
        wlen = pow(root, (mod-1)//length, mod)
        for i in range(0, n, length):
            w = 1
            for j in range(length//2):
                u = a[i+j]
                v = a[i+j+length//2]*w % mod
                a[i+j] = (u+v) % mod
                a[i+j+length//2] = (u-v+mod) % mod
                w = w*wlen % mod
        length *= 2
    return a

def intt(a, mod, root):
    n = len(a)
    inv_root = pow(root, mod-2, mod)
    a_inv = ntt(a, mod, inv_root)
    inv_n = pow(n, mod-2, mod)
    return [(x*inv_n)%mod for x in a_inv]

# ------------------- 正常 Hadamard 矩阵乘法 -------------------
def diag_block_hadamard_matvec(M_blocks, v):
    block_size = len(M_blocks[0])
    k = len(M_blocks)
    n = block_size * k
    y = np.zeros(n, dtype=int)
    for i in range(k):
        y_block = np.zeros(block_size, dtype=int)
        for j in range(k):
            block_idx = (j - i) % k
            block = M_blocks[block_idx]
            v_block = v[j*block_size:(j+1)*block_size]
            y_block += block * v_block
        y[i*block_size:(i+1)*block_size] = y_block
    return y

# ------------------- NTT 注入噪声后的 Hadamard 矩阵乘法 -------------------
def diag_block_hadamard_ntt_noise(M_blocks, v, mod, root):
    block_size = len(M_blocks[0])
    k = len(M_blocks)
    n = block_size * k
    y = np.zeros(n, dtype=int)

    for i in range(k):
        y_block = np.zeros(block_size, dtype=int)
        for j in range(k):
            block_idx = (j - i) % k
            block = M_blocks[block_idx]

            # 向量块
            v_block = v[j*block_size:(j+1)*block_size]

            # 1. NTT
            v_ntt = ntt(list(v_block), mod, root)

            # 2. 注入噪声：随机元素 +1
            noise_idx = np.random.randint(0, block_size)
            v_ntt[noise_idx] = (v_ntt[noise_idx] + 1) % mod

            # 3. INTT
            v_intt = intt(v_ntt, mod, root)

            # 4. Hadamard 乘积 + 累加
            y_block += block * np.array(v_intt)

        y[i*block_size:(i+1)*block_size] = y_block

    return y

# ------------------- 参数设置 -------------------
np.random.seed(0)

n = 1024
block_size = 16
k = n // block_size
mod = 15728641
root = 3

# 对角块矩阵和向量
M_blocks = [np.random.randint(0, mod, size=block_size) for _ in range(k)]
v = np.random.randint(0, mod, size=n)

# ------------------- 计算输出 -------------------
y_normal = diag_block_hadamard_matvec(M_blocks, v)
y_noise = diag_block_hadamard_ntt_noise(M_blocks, v, mod, root)

# ------------------- 可视化 -------------------
plt.figure(figsize=(12,5), dpi=150)

plt.plot(y_normal[:128], 'o-', label='Normal Output', color='dimgrey', lw=2)
plt.plot(y_noise[:128], 'x-', label='Noise-injected Output', color='red')
plt.title("Hadamard Matrix-Vector Output: Normal vs Noise-injected (first 128 elements)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
