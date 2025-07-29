import random
import math

# 生成权重函数（必须为完全平方数长度）
def generate_weights(length):
    assert int(math.sqrt(length))**2 == length, "Length must be a perfect square"
    period = int(math.sqrt(length))
    list1 = [(i % period) + 1 for i in range(length)]
    list2 = [(i // period) + 1 for i in range(length)]
    return [list1[i] + list2[i] for i in range(length)]

# NTT/INTT 基础工具
def bit_reverse(x, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

def modinv(a, mod):
    return pow(a, mod - 2, mod)

def flip_bits(x, rbit, max_bits=None):
    """
    Flip `rbit` random bits in the integer `x`.
    `max_bits` bounds the bit positions (if None, use x.bit_length()).
    """
    if max_bits is None or max_bits == 0:
        max_bits = max(1, x.bit_length())
    for _ in range(rbit):
        pos = random.randrange(0, max_bits)
        x ^= (1 << pos)
    return x

def ntt(a, root, mod):
    n = len(a)
    bits = n.bit_length() - 1
    # bit-reverse
    a = [a[bit_reverse(i, bits)] for i in range(n)]
    length = 2
    while length <= n:
        # ←—— 这里改成 n//length ——↓
        wlen = pow(root, n//length, mod)
        for start in range(0, n, length):
            w = 1
            half = length // 2
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half] * w % mod
                a[start + j]       = (u + v) % mod
                a[start + j + half] = (u - v) % mod
                w = w * wlen % mod
        length <<= 1
    return a

def ntt_with_flip(a, root, mod, t, rbit):
    """
    Perform NTT after randomly perturbing `t` elements of `a`
    by flipping `rbit` bits in each selected element.
    """
    n = len(a)
    # Copy a to avoid in-place changes
    a_pert = a.copy()
    # Choose t distinct indices to perturb
    for idx in random.sample(range(n), min(t, n)):
        # Flip rbit bits within the bit-width of mod
        a_pert[idx] = flip_bits(a_pert[idx] % mod, rbit, max_bits=mod.bit_length())
        a_pert[idx] %= mod
    # Now perform standard NTT on the perturbed array
    return ntt(a_pert, root, mod)

def intt(a, root, mod):
    # 逆变换也用相同的指数形式
    n = len(a)
    inv_n = pow(n, mod-2, mod)
    inv_root = pow(root, mod-2, mod)
    a = ntt(a, inv_root, mod)
    return [(x * inv_n) % mod for x in a]

# Negacyclic NTT/INTT（DWT 风格）
def negacyclic_ntt(a, psi, mod):
    n = len(a)
    # 1) 预加权 psi^i
    a_pw = [(a[i] * pow(psi, i, mod)) % mod for i in range(n)]
    # 2) 标准 NTT，用 psi^2
    root = pow(psi, 2, mod)
    return ntt(a_pw, root, mod)

def negacyclic_ntt_with_flip(a, psi, mod):
    n = len(a)
    # 1) 预加权 psi^i
    a_pw = [(a[i] * pow(psi, i, mod)) % mod for i in range(n)]
    # 2) 标准 NTT，用 psi^2
    root = pow(psi, 2, mod)
    return ntt_with_flip(a_pw, root, mod, 1, 1)

def negacyclic_intt(A, psi, mod):
    n = len(A)
    # 1) 逆 NTT
    root = pow(psi, 2, mod)
    inv_A = intt(A, root, mod)
    # 2) 后加权 psi^{-i}
    psi_inv = modinv(psi, mod)
    return [(inv_A[i] * pow(psi_inv, i, mod)) % mod for i in range(n)]

# 朴素负环多项式卷积（mod x^n + 1）
def poly_mul_naive_negacyclic(a, b, mod):
    n = len(a)
    res = [0] * n
    for i in range(n):
        for j in range(n):
            k = (i + j) % n
            sign = mod - 1 if (i + j) >= n else 1
            res[k] = (res[k] + a[i] * b[j] * sign) % mod
    return res

# NTT 方式负环多项式卷积
def poly_mul_negacyclic_ntt(a, b, psi, mod):
    A_hat = negacyclic_ntt(a, psi, mod)
    B_hat = negacyclic_ntt(b, psi, mod)
    C_hat = [(A_hat[i] * B_hat[i]) % mod for i in range(len(a))]
    return negacyclic_intt(C_hat, psi, mod)

# ECC 校验：比较原始 checksum 与变换后 checksum_hat
def ecc_check(a, psi, mod):
    n = len(a)
    w = generate_weights(n)
    # 预加权 w_pre = D^{-1} w
    psi_inv = modinv(psi, mod)
    w_pre = [(w[i] * pow(psi_inv, i, mod)) % mod for i in range(n)]
    # 计算 w_hat
    root = pow(psi, 2, mod)
    w_hat = intt(w_pre, root, mod)
    # 计算 a_hat
    a_hat = negacyclic_ntt_with_flip(a, psi, mod)
    # 原始和变换后校验和
    checksum      = sum(w[i]     * a[i]     for i in range(n)) % mod
    checksum_hat  = sum(w_hat[i] * a_hat[i] for i in range(n)) % mod
    return checksum, checksum_hat, checksum == checksum_hat

# 主测试
if __name__ == "__main__":
    # 参数：n 为完全平方数且为 2 的幂
    n   = 16
    mod = 97
    psi = 19  # 9 是 mod=17 下的原 8 次单位根

    # n   = 64
    # mod = 257
    # psi = 9   # 原 128 次单位根 mod 257
    epoches = 1000

    collusion = 0
    success = 0
    for _ in range(20000):
        # 随机生成两个多项式 a, b
        a = [random.randint(0, mod-1) for _ in range(n)]
        # b = [random.randint(0, mod-1) for _ in range(n)]
    
        # # —— 校验一：多项式乘法正确性 ——
        # c_naive = poly_mul_naive_negacyclic(a, b, mod)
        # c_ntt   = poly_mul_negacyclic_ntt(a, b, psi, mod)
        # poly_ok = (c_naive == c_ntt)
    
        # —— 校验二：NTT 过程 ECC 校验 ——
        checksum_a, checksum_hat_a, ok_a = ecc_check(a, psi, mod)
        if checksum_a == checksum_hat_a:
            collusion += 1
        else:
            success += 1

        # checksum_b, checksum_hat_b, ok_b = ecc_check(b, psi, mod)
    
        # 输出
        # print(f"a             = {a}")
        # print(f"b             = {b}")
        # print(f"Naive product = {c_naive}")
        # print(f"NTT product   = {c_ntt}")
        # print(f"Polynomial match: {poly_ok}\n")
    
        # print("ECC checksum for a:", checksum_a, "->", checksum_hat_a, "pass?", ok_a)
        # print("ECC checksum for b:", checksum_b, "->", checksum_hat_b, "pass?", ok_b)
    print("Collusion: ", collusion)
    print("Success: ", success)
