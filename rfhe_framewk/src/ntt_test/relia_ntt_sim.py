# relia_ntt_sim.py
# Strict Four-Step NTT with Algorithm-3-style protection:
# - Batch-1 aggregated weighted check over size-n2 sub-NTTs
# - Twiddle (element-wise) weighted linear check
# - Batch-2 aggregated weighted check over size-n1 sub-NTTs
# Fault injection: exactly one modular multiplication output per trial
# Types: SBF (single-bit flip), DBF (double-bit flip), MOF1 (randomize value)
# Modulus: 30-bit NTT-friendly prime q = 3*2^30 + 1

import argparse, math, random
from typing import Dict, Tuple, List
import numpy as np

# ---------- modulus and primitive root ----------
# q = 3221225473          # 3 * 2^30 + 1 (30-bit prime + 1), common NTT-friendly modulus
# g = 5                   # a primitive root modulo q (works for power-of-two NTT sizes)
q = 577
g = 5

# ---------- helpers ----------
def mod_pow(a: int, e: int, m: int = q) -> int:
    return pow(a, e, m)

def root_of_unity(N: int) -> int:
    # requires N | (q-1)
    assert (q - 1) % N == 0, "N must divide q-1"
    return mod_pow(g, (q - 1) // N, q)

BITS = q.bit_length()

def flip_bit_val(x: int, b: int) -> int:
    return (x ^ (1 << b)) % q

def flip_two_bits_val(x: int, b1: int, b2: int) -> int:
    if b1 == b2: return flip_bit_val(x, b1)
    return (x ^ (1 << b1) ^ (1 << b2)) % q

def inject_one(val: int, kind: str) -> int:
    if kind == "SBF":
        b = random.randrange(BITS)         # bit index < bitlen(q)
        return flip_bit_val(val, b)
    if kind == "DBF":
        b1 = random.randrange(BITS); b2 = random.randrange(BITS)
        return flip_two_bits_val(val, b1, b2)
    if kind == "MOF1":
        return random.randrange(q)
    raise ValueError("unknown fault kind")

# ---------- NTT (iterative Cooley–Tukey, DIT) with per-multiply injection hook ----------
def ntt_inplace(vec: List[int], root: int,
                inj_plan: Dict[int, Tuple[str]] = None,
                op_base_idx: int = 0) -> int:
    """
    In-place forward NTT (length = power of two).
    For each twiddle multiplication v = A[j+half] * w, if op_index in inj_plan, corrupt v.
    Returns the next op_index after finishing.
    """
    A = vec
    n = len(A)

    # bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit; bit >>= 1
        j ^= bit
        if i < j:
            A[i], A[j] = A[j], A[i]

    op_idx = op_base_idx
    length = 2
    while length <= n:
        wlen = mod_pow(root, n // length, q)
        for i in range(0, n, length):
            w = 1
            half = length // 2
            for j in range(i, i + half):
                u = A[j]
                v = (A[j + half] * w) % q
                if inj_plan and op_idx in inj_plan:
                    kind = inj_plan[op_idx][0]
                    v = inject_one(v, kind)
                op_idx += 1
                A[j]         = (u + v) % q
                A[j + half]  = (u - v) % q
                w = (w * wlen) % q
        length <<= 1
    return op_idx

def twiddle_mul_inplace(M: np.ndarray, T: np.ndarray,
                        inj_plan: Dict[int, Tuple[str]] = None,
                        op_base_idx: int = 0) -> int:
    """
    Element-wise twiddle multiplication with per-element injection hook.
    """
    n2, n1 = M.shape
    op_idx = op_base_idx
    for r in range(n2):
        Tr = T[r]
        Mr = M[r]
        for c in range(n1):
            v = (int(Mr[c]) * int(Tr[c])) % q
            if inj_plan and op_idx in inj_plan:
                kind = inj_plan[op_idx][0]
                v = inject_one(v, kind)
            op_idx += 1
            Mr[c] = v
    return op_idx

# ---------- protection checks (Algorithm 3 style) ----------
def batch_check_cols(A: np.ndarray, B: np.ndarray, n2_root: int) -> bool:
    """
    Batch-1 check over size-n2 sub-NTTs (columns):
      Let s_in  = sum over columns of A  (vector length n2)
          s_out = sum over columns of B  (vector length n2)
      Choose random w in Z_q^{n2}, compute w_hat = NTT_{n2}(w).
      Check   <w_hat, s_in> == <w, s_out>   mod q.
    """
    s_in  = np.sum(A, axis=1) % q
    s_out = np.sum(B, axis=1) % q
    w = [random.randrange(q) for _ in range(len(s_in))]
    w_hat = w.copy()
    ntt_inplace(w_hat, n2_root, None, 0)
    lhs = int(np.dot(w_hat, s_in) % q)
    rhs = int(np.dot(w,     s_out) % q)
    return lhs == rhs

def twiddle_check(B_before: np.ndarray, B_after: np.ndarray, T: np.ndarray) -> bool:
    """
    Twiddle weighted linear check:
      For random phi in Z_q^{n2}, check  sum_c phi^T (B_before[:,c] .* T[:,c])
                                     == sum_c (phi .* T[:,c])^T B_before[:,c]
    """
    n2, n1 = B_before.shape
    phi = np.array([random.randrange(q) for _ in range(n2)], dtype=np.int64)
    lhs = 0; rhs = 0
    for c in range(n1):
        Tb = (B_before[:, c] * T[:, c]) % q
        lhs = (lhs + int(np.dot(phi, Tb) % q)) % q
        rhs = (rhs + int(np.dot((phi * T[:, c]) % q, B_before[:, c]) % q)) % q
    return lhs == rhs and np.all((B_after % q) == ( (B_before * T) % q ))

def batch_check_rows(B: np.ndarray, C: np.ndarray, n1_root: int) -> bool:
    """
    Batch-2 check over size-n1 sub-NTTs (rows):
      Let r_in  = sum over rows of B  (vector length n1)
          r_out = sum over rows of C  (vector length n1)
      Choose random w in Z_q^{n1}, compute w_hat = NTT_{n1}(w).
      Check   <w_hat, r_in> == <w, r_out>  mod q.
    """
    r_in  = np.sum(B, axis=0) % q
    r_out = np.sum(C, axis=0) % q
    w = [random.randrange(q) for _ in range(len(r_in))]
    w_hat = w.copy()
    ntt_inplace(w_hat, n1_root, None, 0)
    lhs = int(np.dot(w_hat, r_in) % q)
    rhs = int(np.dot(w,     r_out) % q)
    return lhs == rhs

# ---------- four-step NTT with protection and single-op fault injection ----------
def four_step_ntt_protected(a: List[int], inj_plan: Dict[int, Tuple[str]] = None) -> Tuple[List[int], Dict]:
    """
    Reshape to (n2 x n1): columns are size-n2 sub-NTTs in batch-1.
    Steps: batch-1 NTT (cols) -> twiddle -> batch-2 NTT (rows).
    Perform Algorithm-3 checks at each batch and at twiddle.
    Return (flattened output, info dict with booleans and total_ops).
    """
    N = len(a)
    n1 = int(round(math.isqrt(N))); n2 = n1
    assert n1 * n2 == N and (n1 & (n1 - 1)) == 0, "N must be (2^k)^2"

    wN  = root_of_unity(N)
    w_n1 = root_of_unity(n1)
    w_n2 = root_of_unity(n2)

    # Column-major reshape: A[r, c] = a[c*n2 + r]
    A = np.array([a[c*n2 + r] for c in range(n1) for r in range(n2)],
                 dtype=np.int64).reshape(n1, n2).T  # shape (n2, n1)

    # ----- Batch 1: NTT size n2 on each column -----
    B = A.copy()
    op_idx = 0
    for c in range(n1):
        col = B[:, c].tolist()
        op_idx = ntt_inplace(col, w_n2, inj_plan, op_idx)
        B[:, c] = np.array(col, dtype=np.int64)

    ok_batch1 = batch_check_cols(A, B, w_n2)

    # ----- Twiddle multiply -----
    # T[r, c] = wN^(r*c)
    T = np.empty((n2, n1), dtype=np.int64)
    for r in range(n2):
        wr = mod_pow(wN, r, q)
        val = 1
        for c in range(n1):
            T[r, c] = val
            val = (val * wr) % q

    B_before = B.copy()
    op_idx = twiddle_mul_inplace(B, T, inj_plan, op_idx)
    ok_twiddle = twiddle_check(B_before, B, T)

    # ----- Batch 2: NTT size n1 on each row -----
    C = B.copy()
    for r in range(n2):
        row = C[r, :].tolist()
        op_idx = ntt_inplace(row, w_n1, inj_plan, op_idx)
        C[r, :] = np.array(row, dtype=np.int64)

    ok_batch2 = batch_check_rows(B, C, w_n1)

    out = C.T.flatten().astype(int).tolist()
    info = dict(batch1_ok=int(ok_batch1), twiddle_ok=int(ok_twiddle),
                batch2_ok=int(ok_batch2), total_ops=op_idx,
                detected = int(not (ok_batch1 and ok_twiddle and ok_batch2)))
    return out, info

# ---------- experiments ----------
def make_single_injection_plan(total_ops: int, kind: str) -> Dict[int, Tuple[str]]:
    i = random.randrange(total_ops)
    return {i: (kind,)}

def run_trials(N: int, trials: int, kind: str, seed: int = 0) -> Tuple[float, float]:
    random.seed(seed)
    np.random.seed(seed)
    # Dry run to learn op count
    a0 = [0] * N
    _, info0 = four_step_ntt_protected(a0, inj_plan={})
    total_ops = info0["total_ops"]

    detected = 0
    for _ in range(trials):
        a = [random.randrange(q) for _ in range(N)]
        plan = make_single_injection_plan(total_ops, kind)
        _, info = four_step_ntt_protected(a, inj_plan=plan)
        if info["detected"]:
            detected += 1
    det_rate = detected / trials
    miss_rate = 1.0 - det_rate
    return det_rate, miss_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1024, help="NTT length, must be (2^k)^2")
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--kinds", type=str, default="SBF,DBF,MOF1",
                    help="comma-separated: SBF,DBF,MOF1")
    args = ap.parse_args()

    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]
    for k in kinds:
        det, miss = run_trials(args.N, args.trials, k, args.seed)
        print(f"{k}: detect={det:.6f}  miss={miss:.6f}")

if __name__ == "__main__":
    main()
