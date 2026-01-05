# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy

def fft(a):
    """Iterative Cooley-Tukey FFT, in-place"""
    n = len(a)
    j = 0
    # bit-reversal permutation
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
        ang = -2j * np.pi / length
        for i in range(0, n, length):
            w = 1
            for j in range(length // 2):
                u = a[i + j]
                v = a[i + j + length // 2] * w
                a[i + j] = u + v
                a[i + j + length // 2] = u - v
                w *= np.exp(ang)
        length *= 2
    return a

def ifft(a):
    """Inverse FFT"""
    n = len(a)
    a_conj = [x.conjugate() for x in a]
    y = fft(a_conj)
    return [x.conjugate()/n for x in y]

# ------------------- 参数设置 -------------------
n = 1024

# 使用规律性输入
a = np.array([i % 16 for i in range(n)], dtype=complex)

# 原始FFT
A = fft(list(a.copy()))

# 单点噪声
a_noisy = a.copy()
a_noisy[10] += 1
A_noisy = fft(list(a_noisy.copy()))

# ------------------- 可视化 -------------------
plt.figure(figsize=(14,5))

# 左图：输入序列
plt.subplot(1,2,1)
plt.plot(a[:128].real, 'o-', label='Original Input')
plt.plot(a_noisy[:128].real, 'x-', label='Noisy Input')
plt.title("FFT Input (first 128 elements)")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 右图：FFT输出幅度
plt.subplot(1,2,2)
plt.plot(np.abs(A[:128]), 'o-', label='Original FFT')
plt.plot(np.abs(A_noisy[:128]), 'x-', label='Noisy FFT')
plt.title("FFT Output Magnitude (first 128 elements)")
plt.xlabel("Index")
plt.ylabel("Magnitude")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
