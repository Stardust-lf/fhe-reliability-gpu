import random
import math
from typing import List
import matplotlib.pyplot as plt

# ============================================================
# 数论工具函数
# ============================================================

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    witnesses = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in witnesses:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                composite = False
                break
        if composite:
            return False
    return True

def next_prime(cand: int) -> int:
    if cand <= 2:
        return 2
    if cand % 2 == 0:
        cand += 1
    while not is_prime(cand):
        cand += 2
    return cand

def generate_crt_primes(limbs: int, bitwidth: int) -> List[int]:
    primes = set()
    low = 1 << (bitwidth - 1)
    high = (1 << bitwidth) - 1
    while len(primes) < limbs:
        cand = random.randint(low, high) | 1
        p = next_prime(cand)
        if p.bit_length() == bitwidth:
            primes.add(p)
    return list(primes)

def mod_inv(a: int, m: int) -> int:
    return pow(a, -1, m)

# ============================================================
# RNS 表示生成
# ============================================================

def values_to_rns(values: List[int], moduli: List[int]) -> List[List[int]]:
    return [[v % p for v in values] for p in moduli]

def base_conv_fixed(residue_arrays: List[List[int]], moduli_in: List[int], moduli_out: List[int]) -> List[List[int]]:
    m = len(moduli_in)
    N = len(residue_arrays[0])
    P = math.prod(moduli_in)
    
    hat_p = [P // p for p in moduli_in]
    inv_hat_p = [mod_inv(hat_p[j] % moduli_in[j], moduli_in[j]) for j in range(m)]
    
    result_by_element = []
    for i in range(N):
        x = 0
        for j in range(m):
            x += residue_arrays[j][i] * hat_p[j] * inv_hat_p[j]
        x = x % P
        result_by_element.append([x % q for q in moduli_out])
    
    return [[result_by_element[i][k] for i in range(N)] for k in range(len(moduli_out))]

# ============================================================
# 绘图示例（3x3子图）
# ============================================================

def plot_rns_3x3_30bit():
    limbs = 3
    bitwidth = 30
    N = 20  # 向量长度
    
    # 生成输入基和输出基
    moduli_in = generate_crt_primes(limbs, bitwidth)
    moduli_out = generate_crt_primes(limbs, bitwidth)
    
    print("输入基:", moduli_in)
    print("输出基:", moduli_out)
    
    # 生成随机向量（小于输入基积）
    P_in = math.prod(moduli_in)
    values = [random.randint(0, P_in - 1) for _ in range(N)]
    # values = [ for _ in range(N)]
    
    # 输入基 RNS
    residues_in = values_to_rns(values, moduli_in)
    
    # 输出基 RNS
    residues_out = base_conv_fixed(residues_in, moduli_in, moduli_out)
    
    # 带噪声：moduli_in 第一个素数，第5个元素 +1
    residues_in_noisy = [r[:] for r in residues_in]
    residues_in_noisy[0][4] = (residues_in_noisy[0][4] + 1) % moduli_in[0]
    residues_out_noisy = base_conv_fixed(residues_in_noisy, moduli_in, moduli_out)
    
    # 绘制 3x3 子图
    fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    datasets = [residues_in, residues_out, residues_out_noisy]
    titles = ['输入基 RNS', '输出基 RNS', '带噪声输出基 RNS']
    all_moduli = [moduli_in, moduli_out, moduli_out]
    
    for col in range(3):
        for row in range(3):
            axs[row, col].plot(range(N), datasets[col][row], 'o-', label=f'mod {all_moduli[col][row]}')
            axs[row, col].set_ylabel(f'mod {all_moduli[col][row]}')
            axs[row, col].legend()
            axs[row, col].grid(True)
            if row == 2:
                axs[row, col].set_xlabel('元素索引')
        axs[0, col].set_title(titles[col])
    
    plt.tight_layout()
    plt.show()

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    plot_rns_3x3_30bit()
