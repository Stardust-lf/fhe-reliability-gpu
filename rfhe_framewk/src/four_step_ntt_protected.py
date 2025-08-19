#!/usr/bin/env python3
import argparse, math, random
import numpy as np

# ===================== CLI =====================
def parse_args():
    ap = argparse.ArgumentParser(description="FHE fault-injection Monte Carlo with configurable prime bit-length and fold width.")
    ap.add_argument("--pbits", type=int, default=30, help="bit-length for prime modulus P (e.g., 30)")
    ap.add_argument("--W", type=int, default=4, help="fold width W for element-wise stage; M=2^W-1")
    ap.add_argument("--N", type=int, default=64, help="pipeline size N=S*S (S=int(sqrt(N)))")
    ap.add_argument("--trials", type=int, default=10000, help="number of Monte Carlo trials")
    ap.add_argument("--seed", type=int, default=42, help="PRNG seed")
    ap.add_argument("--p", type=int, default=None, help="override prime modulus P (bypasses search if provided)")
    return ap.parse_args()

# ===================== Prime utils =====================
def _is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for 64-bit; probabilistic for larger."""
    if n < 2:
        return False
    small_primes = [2,3,5,7,11,13,17,19,23,29]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # write n-1 = d * 2^s
    d = n - 1
    s = (d & -d).bit_length() - 1 if d else 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases: sufficient for n < 2^64
    if n.bit_length() <= 64:
        bases = [2, 3, 5, 7, 11, 13, 17]
    else:
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]  # probabilistic
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        skip = False
        for _ in range(s-1):
            x = (x * x) % n
            if x == n-1:
                skip = True
                break
        if skip:
            continue
        return False
    return True

def find_prime_with_bitlen(bitlen: int, rng: random.Random) -> int:
    """Find an odd prime of exact bit-length 'bitlen'."""
    if bitlen < 2:
        raise ValueError("bitlen must be >= 2")
    while True:
        # ensure exact bit length and odd
        n = (1 << (bitlen - 1)) | rng.getrandbits(bitlen - 1) | 1
        if _is_probable_prime(n):
            return n

# ===================== Checksums & helpers =====================
def _sum_mod_iter(arr, mod):
    """Safe modular sum using Python int to avoid int64 overflow."""
    acc = 0
    for v in arr:
        acc = (acc + int(v)) % mod
    return acc

def _dot_mod(u, v, mod):
    """Safe modular dot using Python int to avoid int64 overflow."""
    acc = 0
    for a, b in zip(u, v):
        acc = (acc + int(a) * int(b)) % mod
    return acc

def matmul_with_protection(A, B, C=None, mod=2**31-1):
    """
    Scalar checksum equality for matrix multiply:
      sum(C) == col_sums(A)^T * row_sums(B)  (mod)
    If C is None, compute C = A @ B (mod).
    """
    if C is None:
        C = (A @ B) % mod
    col = (np.sum(A, axis=0, dtype=np.int64)) % mod
    row = (np.sum(B, axis=1, dtype=np.int64)) % mod
    lhs = _dot_mod(col, row, mod)
    rhs = _sum_mod_iter(C.reshape(-1), mod)
    return C, (lhs == rhs)

def fold_mod(x, w, M):
    """Compute folding checksum of integer x modulo (2^w - 1)."""
    mask = (1 << w) - 1
    s = 0
    x = int(x)
    while x:
        s += (x & mask)
        x >>= w
    return s % M

def elementwise_with_fold(X, T, W, M, Y=None):
    """
    Element-wise multiply protection under modulus M=2^W-1:
      S_in = sum( fold(X_i) * fold(T_i) ) mod M
      Y    = (X * T) mod M
      S_out= sum( fold(Y_i) ) mod M
      Check S_in == S_out
    If Y provided, verify against Y; else compute Y.
    """
    S_in = 0
    for xi, ti in zip(X, T):
        S_in = (S_in + fold_mod(xi, W, M) * fold_mod(ti, W, M)) % M
    if Y is None:
        Y = (X * T) % M
    S_out = 0
    for yi in Y:
        S_out = (S_out + fold_mod(yi, W, M)) % M
    return Y, (S_in == S_out)

# ===================== Fault models (SCF / MCF) =====================
def inject_scf_matrix(C, subtype, bitwidth, mod, rng):
    """SCF-BF / SCF-MBU on a matrix element."""
    R = C.astype(np.int64).copy()
    rows, cols = R.shape
    i, j = rng.randrange(rows), rng.randrange(cols)

    def wrap(x): return int(x) % mod

    if subtype == "SCF-BF":
        b = rng.randrange(bitwidth)
        R[i, j] = wrap(int(R[i, j]) ^ (1 << b))
    elif subtype == "SCF-MBU":
        K = rng.choice([2, 3, 4]) if bitwidth >= 4 else 2
        start = rng.randrange(max(1, bitwidth - K + 1))
        mask = ((1 << K) - 1) << start
        R[i, j] = wrap(int(R[i, j]) ^ mask)
    return R

def inject_scf_vector(x, subtype, bitwidth, mod, rng):
    """SCF-BF / SCF-MBU on a vector element."""
    y = x.astype(np.int64).copy()
    n = y.shape[0]
    idx = rng.randrange(n)

    def wrap(v): return int(v) % mod

    if subtype == "SCF-BF":
        b = rng.randrange(bitwidth)
        y[idx] = wrap(int(y[idx]) ^ (1 << b))
    elif subtype == "SCF-MBU":
        K = 2 if bitwidth < 3 else rng.choice([2, 3])
        start = rng.randrange(max(1, bitwidth - K + 1))
        mask = ((1 << K) - 1) << start
        y[idx] = wrap(int(y[idx]) ^ mask)
    return y

def inject_mcf_matrix(A, B, C, subtype, bitwidth, mod, rng):
    """MCF-PPE / MCF-CTE / MCF-CLE on matmul result."""
    R = C.astype(np.int64).copy()
    rows, cols = R.shape

    def wrap(x): return int(x) % mod

    if subtype == "MCF-PPE":
        i, j = rng.randrange(rows), rng.randrange(cols)
        b = rng.randrange(bitwidth)
        R[i, j] = wrap(int(R[i, j]) ^ (1 << b))

    elif subtype == "MCF-CTE":
        i, j = rng.randrange(rows), rng.randrange(cols)
        b = rng.randrange(max(1, bitwidth // 2), bitwidth)  # bias to high bits
        R[i, j] = wrap(int(R[i, j]) ^ (1 << b))

    elif subtype == "MCF-CLE":
        i = rng.randrange(rows)
        k0 = rng.randrange(B.shape[0])
        s = rng.choice([+1, -1])
        mfac = rng.choice([1, 2])
        aik = int(A[i, k0])
        for j in range(cols):
            delta = s * mfac * aik * int(B[k0, j])
            R[i, j] = wrap(int(R[i, j]) + delta)

    return R

def inject_mcf_vector(X, T, Y, subtype, bitwidth, mod, rng):
    """MCF-PPE / MCF-CTE / MCF-CLE on element-wise result."""
    R = Y.astype(np.int64).copy()
    n = R.shape[0]
    idx = rng.randrange(n)

    def wrap(x): return int(x) % mod

    if subtype == "MCF-PPE":
        b = rng.randrange(bitwidth)
        R[idx] = wrap(int(R[idx]) ^ (1 << b))

    elif subtype == "MCF-CTE":
        b = rng.randrange(max(1, bitwidth - 2), bitwidth)  # top 1-2 bits
        R[idx] = wrap(int(R[idx]) ^ (1 << b))

    elif subtype == "MCF-CLE":
        k = rng.choice([-1, 2, -2])
        wrong = (int(X[idx]) * int(T[idx])) % mod
        R[idx] = wrap(k * wrong)

    return R

# ===================== Pipeline =====================
FAULT_TYPES = {
    "SCF-BF":  "storage_single_bit",
    "SCF-MBU": "storage_multi_bit",
    "MCF-PPE": "mul_partial_product",
    "MCF-CTE": "mul_carry_tree",
    "MCF-CLE": "mul_control_logic",
}

def run_one_trial(ftype, S, P, W, M, rng):
    """
    3-stage pipeline:
      Stage1: C1 = (A1 @ B1) mod P, protected by column-sum · row-sum
      Stage2: Y  = (vec(C1) * T) mod M, protected by fold checksum
      Stage3: C3 = (A2 @ B2) mod P, protected by column-sum · row-sum
               where B2 = reshape(Y % P, S x S)
    Inject exactly ONE fault at a random stage, following ftype.
    """
    bitwidth_p = P.bit_length()

    # Stage 1 data
    A1 = np.random.randint(0, P, size=(S, S), dtype=np.int64)
    B1 = np.random.randint(0, P, size=(S, S), dtype=np.int64)
    C1 = (A1 @ B1) % P

    # Stage 2 data
    X  = (C1.reshape(-1).astype(np.int64)) % M
    T  = np.random.randint(0, M, size=S*S, dtype=np.int64)
    Y  = (X * T) % M

    # Stage 3 data
    B2 = (Y.reshape(S, S).astype(np.int64)) % P
    A2 = np.random.randint(0, P, size=(S, S), dtype=np.int64)
    C3 = (A2 @ B2) % P

    inject_stage = rng.randrange(1, 4)
    det1 = det2 = det3 = False

    # Stage 1 protection
    if inject_stage == 1:
        if ftype.startswith("SCF"):
            C1f = inject_scf_matrix(C1, ftype, bitwidth_p, P, rng)
        else:
            C1f = inject_mcf_matrix(A1, B1, C1, ftype, bitwidth_p, P, rng)
        _, ok1 = matmul_with_protection(A1, B1, C=C1f, mod=P)
        det1 = (ok1 is False)
    else:
        _, _ = matmul_with_protection(A1, B1, C=C1, mod=P)

    # Stage 2 protection
    if inject_stage == 2:
        if ftype.startswith("SCF"):
            Yf = inject_scf_vector(Y, ftype, W, M, rng)
        else:
            Yf = inject_mcf_vector(X, T, Y, ftype, W, M, rng)
        _, ok2 = elementwise_with_fold(X, T, W, M, Y=Yf)
        det2 = (ok2 is False)
    else:
        _, _ = elementwise_with_fold(X, T, W, M, Y=Y)

    # Stage 3 protection
    if inject_stage == 3:
        if ftype.startswith("SCF"):
            C3f = inject_scf_matrix(C3, ftype, bitwidth_p, P, rng)
        else:
            C3f = inject_mcf_matrix(A2, B2, C3, ftype, bitwidth_p, P, rng)
        _, ok3 = matmul_with_protection(A2, B2, C=C3f, mod=P)
        det3 = (ok3 is False)
    else:
        _, _ = matmul_with_protection(A2, B2, C=C3, mod=P)

    return det1, det2, det3, inject_stage

def monte_carlo(ftype, trials, S, P, W, M, rng):
    inj_counts = [0, 0, 0]
    undetected = [0, 0, 0]
    for _ in range(trials):
        det1, det2, det3, inj_stage = run_one_trial(ftype, S, P, W, M, rng)
        idx = inj_stage - 1
        inj_counts[idx] += 1
        if idx == 0 and det1 is False:
            undetected[idx] += 1
        if idx == 1 and det2 is False:
            undetected[idx] += 1
        if idx == 2 and det3 is False:
            undetected[idx] += 1
    p = [(undetected[i] / inj_counts[i]) if inj_counts[i] > 0 else 0.0 for i in range(3)]
    return inj_counts, undetected, p

# ===================== Main =====================
def main():
    args = parse_args()
    rng = random.Random(args.seed)

    # Derive S from N
    S = int(math.isqrt(args.N))
    assert S * S == args.N, "N must be a perfect square (N=S*S)."

    # Build P
    if args.p is not None:
        P = int(args.p)
        if P <= 2 or not _is_probable_prime(P):
            raise ValueError("Provided --p is not prime.")
        if P.bit_length() != args.pbits:
            # Not fatal, but warn via print
            print(f"[warn] --p bit-length {P.bit_length()} != --pbits {args.pbits}")
    else:
        P = find_prime_with_bitlen(args.pbits, rng)
    bitwidth_p = P.bit_length()

    # Build M from W
    if args.W < 2:
        raise ValueError("--W must be >= 2")
    W = args.W
    M = (1 << W) - 1

    print(f"[config] Pbits={bitwidth_p}, P={P}")
    print(f"[config] W={W}, M=2^W-1={M}")
    print(f"[config] N={args.N} => S={S}, trials={args.trials}, seed={args.seed}")

    for ftype in ["SCF-BF", "SCF-MBU", "MCF-PPE", "MCF-CTE", "MCF-CLE"]:
        inj_counts, undetected, p_stage = monte_carlo(ftype, args.trials, S, P, W, M, rng)
        print(f"[{ftype} - {FAULT_TYPES[ftype]}]")
        for s, p in enumerate(p_stage, start=1):
            print(f"  stage{s}: injected={inj_counts[s-1]}, undetected={undetected[s-1]}, collision_prob={p:.6f}")

if __name__ == "__main__":
    main()
