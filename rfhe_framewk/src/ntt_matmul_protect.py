# stage1_unittest_cli.py
# python stage1_unittest_cli.py
import math, random, sys
from tqdm import trange
import numpy as np

rng = random.Random(12345)

# ---------- Miller-Rabin（< 2^64 确定性底集） ----------
def _is_probable_prime(n:int) -> bool:
    if n < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n == p: return True
        if n % p == 0: return False
    # write n-1 = d*2^s
    d, s = n-1, 0
    while d % 2 == 0:
        d //= 2; s += 1
    # deterministic bases for < 2^64
    for a in [2,3,5,7,11,13,17]:
        if a % n == 0: 
            return True
        x = pow(a, d, n)
        if x == 1 or x == n-1: 
            continue
        skip = False
        for _ in range(s-1):
            x = (x*x) % n
            if x == n-1:
                skip = True
                break
        if not skip:
            return False
    return True

def find_prime_with_bits(bits:int) -> int:
    # search downward from 2^bits-1
    start = (1 << bits) - 1
    n = start | 1  # odd
    while n > 2:
        if _is_probable_prime(n):
            return n
        n -= 2
    raise RuntimeError(f"no prime found for bits={bits}")

# ---------- 保护：列和·行和 校验 ----------
def _sum_mod(arr, mod:int) -> int:
    acc = 0
    for v in arr: acc = (acc + int(v)) % mod
    return acc

def _dot_mod(u, v, mod:int) -> int:
    acc = 0
    for a,b in zip(u, v): acc = (acc + int(a)*int(b)) % mod
    return acc

def matmul_with_protection(A, B, C=None, mod:int=0):
    if C is None:
        C = (A @ B) % mod
    col = (np.sum(A, axis=0, dtype=np.int64)) % mod
    row = (np.sum(B, axis=1, dtype=np.int64)) % mod
    lhs = _dot_mod(col, row, mod)
    rhs = _sum_mod(C.reshape(-1), mod)
    return C, (lhs == rhs)

# ---------- 故障注入（矩阵结果端） ----------
def _pick_two_pos(rows, cols):
    allpos = [(r,c) for r in range(rows) for c in range(cols)]
    return rng.sample(allpos, 2)

def inject_fault_matrix(C:np.ndarray, mode:int, bitwidth:int, mod:int) -> np.ndarray:
    R = C.astype(np.int64).copy()
    rows, cols = R.shape

    def wrap(x): return int(x) % mod

    if mode == 1:  # flip_1bit_1elem
        i,j = rng.randrange(rows), rng.randrange(cols)
        b = rng.randrange(bitwidth)
        R[i,j] = wrap(int(R[i,j]) ^ (1 << b))

    elif mode == 2:  # flip_1bit_2elems（不同元素、不同bit）
        (i1,j1),(i2,j2) = _pick_two_pos(rows, cols)
        b1,b2 = rng.sample(range(bitwidth), 2)
        R[i1,j1] = wrap(int(R[i1,j1]) ^ (1 << b1))
        R[i2,j2] = wrap(int(R[i2,j2]) ^ (1 << b2))

    elif mode == 3:  # flip_2bits_1elem（同元素不同bit）
        i,j = rng.randrange(rows), rng.randrange(cols)
        b1,b2 = rng.sample(range(bitwidth), 2)
        R[i,j] = wrap(int(R[i,j]) ^ (1 << b1) ^ (1 << b2))

    elif mode == 4:  # randval_1elem
        i,j = rng.randrange(rows), rng.randrange(cols)
        R[i,j] = wrap(rng.randrange(mod))

    elif mode == 5:  # randval_2elems（不同元素、不同值）
        (i1,j1),(i2,j2) = _pick_two_pos(rows, cols)
        v1, v2 = rng.randrange(mod), rng.randrange(mod)
        while v2 == v1:
            v2 = rng.randrange(mod)
        R[i1,j1] = wrap(v1); R[i2,j2] = wrap(v2)

    return R

# ---------- 单次试验（stage1） ----------
def stage1_trial(P:int, S:int, mode:int) -> bool:
    A = np.random.randint(0, P, size=(S,S), dtype=np.int64)
    B = np.random.randint(0, P, size=(S,S), dtype=np.int64)
    C = (A @ B) % P
    Cf = inject_fault_matrix(C, mode, bitwidth=P.bit_length(), mod=P)
    _, ok = matmul_with_protection(A, B, C=Cf, mod=P)
    return not ok  # True=检测到

ERROR_NAMES = {
    1:"flip_1bit_1elem",
    2:"flip_1bit_2elems",
    3:"flip_2bits_1elem",
    4:"randval_1elem",
    5:"randval_2elems",
}

def run_mc_for_bits(bits_list, trials=1000, N=16):
    S = int(math.isqrt(N))
    assert S*S == N
    print(f"# stage1 Monte Carlo | N={N}, S={S}, trials={trials} per prime\n")
    for bits in bits_list:
        P = find_prime_with_bits(bits)
        print(f"P_bits={bits}, prime={P}")
        for mode in range(1,6):
            detected = 0
            for _ in trange(trials, desc=f"{ERROR_NAMES[mode]}", leave=False):
                detected += stage1_trial(P, S, mode)
            collision = 1 - detected / trials
            print(f"  {ERROR_NAMES[mode]:>18s}  detected={detected:4d}/{trials}  collision={collision:.6e}")
        print()

# ---------- 可选：简单位级单元断言 ----------
def quick_unit_assert(bits=30, trials=2000, N=16):
    P = find_prime_with_bits(bits)
    S = int(math.isqrt(N))
    # 三个bit翻转应全部检出
    for mode in (1,2,3):
        det = sum(stage1_trial(P, S, mode) for _ in range(trials))
        assert det == trials, f"mode {mode} not fully detected under P={P}"
    # 随机化场景碰撞 ~ 1/P，给宽松区间检验（无需严格）
    # 这里只打印，不强行断言
    for mode in (4,5):
        det = sum(stage1_trial(P, S, mode) for _ in range(trials))
        coll = 1 - det/trials
        print(f"[quick-check] P={P} mode={ERROR_NAMES[mode]} collision≈{coll:.3e} (theory≈{1/P:.3e})")

if __name__ == "__main__":
    # 配置：位数清单，可改
    # bits_list = [20, 24, 28, 30, 32]
    bits_list = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # 若从命令行传入位数列表，用它
    if len(sys.argv) > 1:
        bits_list = [int(x) for x in sys.argv[1:]]
    quick_unit_assert(bits=30, trials=2000, N=16)
    run_mc_for_bits(bits_list, trials=1000000, N=16)
