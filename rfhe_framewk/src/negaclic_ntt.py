from tqdm import tqdm
import random
import math
import argparse

FLIP_ELEMENT, FLIP_BIT_PER_ELEMENT = 2, 1
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

def flip_bits(x: int, rbit: int):
    """
    Flip `rbit` random bits in x, return (new_x, bit_positions, original_x).
    """
    original = x
    bit_positions = []
    for _ in range(rbit):
        pos = random.randrange(0, x.bit_length())
        x ^= (1 << pos)
        bit_positions.append(pos)
    return x, bit_positions, original

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
    a_pert = a.copy()
    flips = []
    for idx in random.sample(range(n), min(t, n)):
        orig = a_pert[idx] % mod
        new_val, bits, original = flip_bits(orig, rbit)
        new_val %= mod
        a_pert[idx] = new_val
        flips.append((idx, original, new_val, bits))
    # Now perform standard NTT on the perturbed array
    transformed = ntt(a_pert, root, mod)
    return transformed, flips

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
    return ntt_with_flip(a_pw, root, mod, FLIP_ELEMENT, FLIP_BIT_PER_ELEMENT)

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
    a_hat, flips = negacyclic_ntt_with_flip(a, psi, mod)
    # a_hat = negacyclic_ntt_with_flip(a, psi, mod)
    # 原始和变换后校验和
    checksum      = sum(w[i]     * a[i]     for i in range(n)) % mod
    checksum_hat  = sum(w_hat[i] * a_hat[i] for i in range(n)) % mod
    if checksum == checksum_hat:
        
        print("checksum:", sum(w[i]     * a[i]     for i in range(n)), "checksum_hat:", sum(w_hat[i] * a_hat[i] for i in range(n)), "Modulus:", mod)
        print("Flip IDX:", flips)
    return checksum, checksum_hat, checksum == checksum_hat


def is_prime(n: int, k: int = 10) -> bool:
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    # small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    for sp in small_primes:
        if n == sp:
            return True
        if n % sp == 0:
            return False
    # write n-1 as d*2^s
    d, s = n - 1, 0
    while d & 1 == 0:
        d >>= 1
        s += 1
    def trial(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False
    for _ in range(k):
        a = random.randrange(2, n - 2)
        if not trial(a):
            return False
    return True

def find_prime_with_divisor(bits: int, divisor: int) -> int:
    """Find a random prime of `bits` length with (prime-1) divisible by `divisor`."""
    while True:
        # ensure high bit set and odd
        p = random.getrandbits(bits) | (1 << (bits - 1)) | 1
        if not is_prime(p):
            continue
        if (p - 1) % divisor == 0:
            return p

def factor(n: int):
    """
    Return the list of distinct prime factors of n using trial division.
    """
    primes = []
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            primes.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        primes.append(temp)
    return primes

def find_primitive_root(mod: int) -> int:
    """
    Find a primitive root modulo 'mod' (prime).
    Returns g such that g is a generator of the multiplicative group Z_mod*.
    """
    phi = mod - 1
    factors = factor(phi)
    for g in range(2, mod):
        if all(pow(g, phi // q, mod) != 1 for q in factors):
            return g
    raise ValueError(f"No primitive root found for modulus {mod}")

def find_primitive_kth_root(mod: int, k: int) -> int:
    """
    Find a primitive k-th root of unity modulo 'mod' (prime).
    Requires that k divides mod-1.
    """
    phi = mod - 1
    if phi % k != 0:
        raise ValueError(f"k={k} does not divide mod-1={phi}")
    g = find_primitive_root(mod)
    # candidate for primitive k-th root
    psi = pow(g, phi // k, mod)
    # validate order exactly k
    if pow(psi, k, mod) != 1:
        raise AssertionError("psi^k != 1")
    # check no smaller divisor yields 1
    for d in factor(k):
        if d < k and pow(psi, d, mod) == 1:
            raise AssertionError(f"psi has smaller order divisor {d}")
    return psi

# 主测试
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECC NTT Test Harness")
    parser.add_argument("n", type=int, help="Length (power-of-two perfect square)")
    parser.add_argument("mod_bits", type=int, help="Desired bit-length of prime modulus")
    args = parser.parse_args()

    n = args.n
    mod_bits = args.mod_bits

    # Validate n
    if int(math.isqrt(n))**2 != n or (n & (n - 1)) != 0:
        raise ValueError("n must be a power-of-two perfect square")

    sqrt_n = int(math.isqrt(n))
    k = 2 * sqrt_n  # order of psi

    # Find a prime modulus of desired bit length
    mod = find_prime_with_divisor(mod_bits, k)

    # Compute psi = primitive (2*sqrt(n))-th root of unity
    psi = find_primitive_kth_root(mod, k)

    print(f"n         = {n}")
    print(f"mod       = {mod} ({mod.bit_length()} bits)")
    print(f"psi (ord {k}) = {psi}")
    # epoches = 100000


    collusion = 0
    success = 0
    for _ in tqdm(range(1000000), desc="ECC checking"):
        # 随机生成两个多项式 a, b
        a = [random.randint(0, mod-1) for _ in range(n)]
        # b = [random.randint(0, mod-1) for _ in range(n)]
    
        # # —— 校验一：多项式乘法正确性 ——
        # c_naive = poly_mul_naive_negacyclic(a, b, mod)
        # c_ntt   = poly_mul_negacyclic_ntt(a, b, psi, mod)
        # poly_ok = (c_naive == c_ntt)
    
        # —— 校验二：NTT 过程 ECC 校验 ——
        checksum_a, checksum_hat_a, ok_a = ecc_check(a, psi, mod)
        if ok_a:
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
