import random
import math

def is_prime(p):
    if p <= 3:
        return p == 2 or p == 3
    if p % 2 == 0 or p % 3 == 0:
        return False
    i = 5
    while i * i <= p:
        if p % i == 0 or p % (i + 2) == 0:
            return False
        i += 6
    return True

def find_mod_and_root(n):
    """寻找形如 mod = k * 2^n + 1 的素数和对应的原根"""
    power = 1 << n
    k = 1
    while True:
        mod = k * power + 1
        if is_prime(mod):
            for root in range(2, mod):
                if pow(root, power, mod) == 1 and all(pow(root, power // d, mod) != 1 for d in range(2, power)):
                    return mod, root
        k += 1

def modinv(a, mod):
    return pow(a, mod - 2, mod)

def bit_reverse(x, bits):
    result = 0
    for _ in range(bits):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

def ntt(a, root, mod):
    n = len(a)
    bits = n.bit_length() - 1
    a = [a[bit_reverse(i, bits)] for i in range(n)]
    
    len_ = 2
    while len_ <= n:
        wlen = pow(root, (mod - 1) // len_, mod)
        for i in range(0, n, len_):
            w = 1
            for j in range(len_ // 2):
                u = a[i + j]
                v = a[i + j + len_ // 2] * w % mod
                a[i + j] = (u + v) % mod
                a[i + j + len_ // 2] = (u - v + mod) % mod
                w = w * wlen % mod
        len_ <<= 1
    return a

def intt(a, root, mod):
    n = len(a)
    inv_n = modinv(n, mod)
    inv_root = modinv(root, mod)
    a = ntt(a, inv_root, mod)
    return [(x * inv_n) % mod for x in a]

def poly_mul_ntt(a, b, root, mod):
    n = 1
    while n < len(a) + len(b) - 1:
        n <<= 1
    a_pad = a + [0] * (n - len(a))
    b_pad = b + [0] * (n - len(b))
    
    A = ntt(a_pad, root, mod)
    B = ntt(b_pad, root, mod)
    C = [(x * y) % mod for x, y in zip(A, B)]
    return intt(C, root, mod)

def poly_mul_naive(a, b, mod):
    res = [0] * (len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i + j] = (res[i + j] + a[i] * b[j]) % mod
    return res

def generate_weights(length):
    assert int(math.sqrt(length))**2 == length, "Length must be a perfect square"
    period = int(math.sqrt(length))
    
    list1 = [(i % period) + 1 for i in range(length)]
    list2 = [(i // period) + 1 for i in range(length)]
    
    return [list1[i] + list2[i] for i in range(len(list1))]

def is_symmetric(matrix):
    n = len(matrix)
    return all(matrix[i][j] == matrix[j][i] for i in range(n) for j in range(n))

if __name__ == "__main__":
    # 参数设置
    deg = 64  # 多项式 degree（长度）
    log_n = (2 * deg).bit_length()  # 最小支持长度（下一步2的幂）
    
    # 自动寻找模数和单位根
    mod, root = find_mod_and_root(log_n)
    print(f"Using mod = {mod}, root = {root} for NTT length {1<<log_n}")

    # 随机多项式
    A = [random.randint(0, mod - 1) for _ in range(deg)]
    B = [random.randint(0, mod - 1) for _ in range(deg)]
    print("A(x):", A)
    print("B(x):", B)

    ntt_result = poly_mul_ntt(A, B, root, mod)
    naive_result = poly_mul_naive(A, B, mod)
    print("NTT mul result:   ", ntt_result[:len(naive_result)])
    print("Naive mul result: ", naive_result)

    if ntt_result[:len(naive_result)] == naive_result:
        print("✅ Verified: Results match.")
    else:
        print("❌ Mismatch detected!")

    w = generate_weights(deg)
    w_hat = intt(w, root, mod)
    A_hat = ntt(A, root , mod)

    print("W * A", sum([w[i] * A[i] for i in range(len(w))]) % mod)
    print("W^hat * A^hat", sum([w_hat[i] * A_hat[i] for i in range(len(w_hat))]) % mod)

