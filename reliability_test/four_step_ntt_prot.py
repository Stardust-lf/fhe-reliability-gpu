# four_step_ntt_protected.py
# Full implementation:
# 1) Direct O(N^2) NTT for reference
# 2) Four-step NTT (N = n1 * n2, using n1 = n2 = sqrt(N) here)
# 3) Correct matrix factorization W_N = Pi_k (I ⊗ W_{n1}) S D (I ⊗ W_{n2}) Pi_t
# 4) Protection check for each sqrt(N) x sqrt(N) matrix multiply using:
#    sum(C) == col_sums(A)^T * row_sums(B)
# 5) End-to-end verification and example fault injection
#
# NOTE: This "protection" is only a scalar checksum equality. It is weak and
#       not a side-channel or robust ABFT scheme. Use at your own risk.

import random
import numpy as np

MOD = 998244353
G = 3  # primitive root for MOD

# ---------- Utilities ----------

def powmod(a, e, mod=MOD):
    return pow(a, e, mod)

def root_of_unity(N, mod=MOD, g=G):
    return pow(g, (mod - 1) // N, mod)

def sum_all(M, mod=MOD):
    return int(np.sum(M, dtype=object) % mod)

def col_sums(A, mod=MOD):
    # Sum over rows to get per-column sums
    return np.array([int(np.sum(A[:, k], dtype=object) % mod) for k in range(A.shape[1])], dtype=object)

def row_sums(B, mod=MOD):
    # Sum over columns to get per-row sums
    return np.array([int(np.sum(B[k, :], dtype=object) % mod) for k in range(B.shape[0])], dtype=object)

def dot(u, v, mod=MOD):
    acc = 0
    for i in range(len(u)):
        acc = (acc + int(u[i]) * int(v[i])) % mod
    return acc

def matmul_mod(A, B, mod=MOD):
    return (A @ B) % mod

# ---------- Dense NTT for reference ----------

def ntt_direct(a, N, mod=MOD, g=G):
    """O(N^2) reference NTT: y_k = sum_t a_t * w^(k*t) (mod)."""
    w = root_of_unity(N, mod, g)
    y = [0] * N
    for k in range(N):
        acc = 0
        for t in range(N):
            acc = (acc + a[t] * powmod(w, k * t, mod)) % mod
        y[k] = acc
    return y

def build_ntt_matrix(N, mod=MOD, g=G):
    """Build the dense NTT matrix W_N with entries w^(k*t)."""
    w = root_of_unity(N, mod, g)
    M = np.zeros((N, N), dtype=object)
    for k in range(N):
        for t in range(N):
            M[k, t] = powmod(w, k * t, mod)
    return M % mod

# ---------- Four-step NTT, algorithmic version ----------

def four_step_ntt(a, N, mod=MOD, g=G):
    """Four-step NTT with N = n1*n2, using n1 = n2 = sqrt(N)."""
    n1 = int(N**0.5)
    assert n1 * n1 == N
    n2 = n1
    w = root_of_unity(N, mod, g)
    w_n1 = pow(w, n1, mod)  # order n2
    w_n2 = pow(w, n2, mod)  # order n1

    # Reshape: A[t2][t1] = a[t1 + n1*t2]
    A = [[a[t1 + n1 * t2] for t1 in range(n1)] for t2 in range(n2)]

    # Step 1: For each column t1, do an n2-point NTT along t2 using w^(n1)
    B = [[0] * n2 for _ in range(n1)]  # shape: [t1][k2]
    for t1 in range(n1):
        for k2 in range(n2):
            s = 0
            for t2 in range(n2):
                s = (s + A[t2][t1] * pow(w_n1, k2 * t2, mod)) % mod
            B[t1][k2] = s

    # Step 2: Twiddle multiply by w^(k2 * t1)
    C = [[(B[t1][k2] * pow(w, k2 * t1, mod)) % mod for k2 in range(n2)] for t1 in range(n1)]

    # Step 3: For each fixed k2, do an n1-point NTT along t1 using w^(n2)
    Y = [[0] * n2 for _ in range(n1)]  # shape: [k1][k2]
    for k2 in range(n2):
        for k1 in range(n1):
            s = 0
            for t1 in range(n1):
                s = (s + C[t1][k2] * pow(w_n2, k1 * t1, mod)) % mod
            Y[k1][k2] = s

    # Step 4: Output reorder: y[k1*n2 + k2] = Y[k1][k2]
    y = [0] * N
    for k1 in range(n1):
        for k2 in range(n2):
            y[k1 * n2 + k2] = Y[k1][k2]
    return y

# ---------- Matrix factorization pieces ----------

def build_small_Ws(N, mod=MOD, g=G):
    """Return (Wn1, Wn2, n1, n2) where n1=n2=sqrt(N)."""
    n1 = int(N**0.5); assert n1*n1 == N
    n2 = n1
    wN = root_of_unity(N, mod, g)
    w_n1 = pow(wN, n1, mod)  # order n2
    w_n2 = pow(wN, n2, mod)  # order n1

    Wn1 = np.zeros((n1, n1), dtype=object)
    for r in range(n1):
        for c in range(n1):
            Wn1[r, c] = pow(w_n2, r * c, mod)

    Wn2 = np.zeros((n2, n2), dtype=object)
    for r in range(n2):
        for c in range(n2):
            Wn2[r, c] = pow(w_n1, r * c, mod)

    return Wn1 % mod, Wn2 % mod, n1, n2

def build_four_step_factors_fixed(N, mod=MOD, g=G):
    """Return Pi_k, B, S, D, A, Pi_t for the correct factorization:
       W_N = Pi_k @ (I_{n2} ⊗ W_{n1}) @ S @ D @ (I_{n1} ⊗ W_{n2}) @ Pi_t
    """
    Wn1, Wn2, n1, n2 = build_small_Ws(N, mod, g)
    wN = root_of_unity(N, mod, g)

    I1 = np.identity(n1, dtype=object)
    I2 = np.identity(n2, dtype=object)
    Ntot = N

    # Pi_t: linear t -> (t1, t2) with s = t1*n2 + t2
    Pi_t = np.zeros((Ntot, Ntot), dtype=object)
    for t in range(Ntot):
        t1 = t % n1
        t2 = t // n1
        s = t1 * n2 + t2
        Pi_t[s, t] = 1

    # Stage-1: (I_{n1} ⊗ W_{n2})
    A = np.kron(I1, Wn2) % mod

    # Twiddle D over (t1,k2): diag value wN^(t1*k2)
    D = np.zeros((Ntot, Ntot), dtype=object)
    for t1 in range(n1):
        for k2 in range(n2):
            s = t1 * n2 + k2
            D[s, s] = pow(wN, t1 * k2, mod)

    # Swap to (k2, t1)
    S = np.zeros((Ntot, Ntot), dtype=object)
    for t1 in range(n1):
        for k2 in range(n2):
            s_from = t1 * n2 + k2
            s_to = k2 * n1 + t1
            S[s_to, s_from] = 1

    # Stage-2: (I_{n2} ⊗ W_{n1})
    B = np.kron(I2, Wn1) % mod

    # Pi_k: (k2,k1) -> k1*n2 + k2
    Pi_k = np.zeros((Ntot, Ntot), dtype=object)
    for k1 in range(n1):
        for k2 in range(n2):
            s_from = k2 * n1 + k1
            k = k1 * n2 + k2
            Pi_k[k, s_from] = 1

    return Pi_k % mod, B % mod, S % mod, D % mod, A % mod, Pi_t % mod

# ---------- Protection check for each sqrt x sqrt multiply ----------

def protection_check(A, B, C=None, mod=MOD):
    """Scalar checksum equality:
       sum(C) == col_sums(A)^T * row_sums(B)
       If C is None, compute A@B first.
    """
    if C is None:
        C = matmul_mod(A, B, mod)
    lhs = sum_all(C, mod)
    rhs = dot(col_sums(A, mod), row_sums(B, mod), mod)
    return lhs, rhs, (lhs % mod) == (rhs % mod)

def four_step_with_protection_vector(a, N, mod=MOD, g=G):
    """Run four-step NTT and apply the protection equality to the two
       sqrt(N) x sqrt(N) matrix multiplies that appear in the matrix factorization.
       Returns (y, checks) where checks is a dict with details.
    """
    W = build_ntt_matrix(N, mod, g)
    y_ref = ntt_direct(a, N, mod, g)

    # Build factor matrices
    Pi_k, B, S, D, A, Pi_t = build_four_step_factors_fixed(N, mod, g)

    # Vectorize inputs at each stage to extract the effective small-matrix multiplies:
    x0 = np.array(a, dtype=object).reshape(N, 1) % mod

    # Stage 0: Pi_t
    x1 = (Pi_t @ x0) % mod  # shape N x 1, layout by (t1, t2) with s = t1*n2 + t2

    # Stage 1 multiply: (I_{n1} ⊗ W_{n2}) @ x1
    # We can view this as n1 independent multiplies of W_{n2} by length-n2 vectors.
    # For the protection equality, we form a block matrix A1 and B1 so that C1 = A1 @ B1 matches the block result.
    n1 = int(N**0.5); n2 = n1
    Wn1, Wn2, _, _ = build_small_Ws(N, mod, g)

    # Build A1 (block-diagonal of n1 copies of Wn2), B1 (stacked vectors per block) to apply the equality once.
    # A1 shape: (N x N), B1 shape: (N x 1)
    A1 = np.kron(np.identity(n1, dtype=object), Wn2) % mod
    B1 = x1 % mod
    C1 = (A1 @ B1) % mod  # should equal (I ⊗ Wn2) @ x1
    lhs1, rhs1, ok1 = protection_check(A1, B1, C1, mod)

    x2 = C1  # proceed
    # Stage 1.5: Twiddle D
    x3 = (D @ x2) % mod
    # Stage 1.75: Swap S
    x4 = (S @ x3) % mod

    # Stage 2 multiply: (I_{n2} ⊗ W_{n1}) @ x4
    A2 = np.kron(np.identity(n2, dtype=object), Wn1) % mod
    B2 = x4 % mod
    C2 = (A2 @ B2) % mod
    lhs2, rhs2, ok2 = protection_check(A2, B2, C2, mod)

    x5 = C2
    # Stage 3: Pi_k
    x6 = (Pi_k @ x5) % mod

    y = [int(v % mod) for v in x6.reshape(-1)]

    # Confirm equality to reference NTT
    y_ok = (y == y_ref)

    checks = {
        "stage1": {"lhs": int(lhs1), "rhs": int(rhs1), "ok": bool(ok1)},
        "stage2": {"lhs": int(lhs2), "rhs": int(rhs2), "ok": bool(ok2)},
        "y_matches_reference": bool(y_ok)
    }
    return y, checks

# ---------- Demo and tests ----------

def demo(N=16, seed=0):
    rng = random.Random(seed)
    a = [rng.randrange(MOD) for _ in range(N)]
    y, checks = four_step_with_protection_vector(a, N)
    print("Checks:", checks)

    # Fault injection after stage-1 multiply to see detection
    # Re-run internals manually for demonstration
    Pi_k, B, S, D, A, Pi_t = build_four_step_factors_fixed(N)
    n1 = int(N**0.5); n2 = n1
    Wn1, Wn2, _, _ = build_small_Ws(N)

    x0 = np.array(a, dtype=object).reshape(N, 1) % MOD
    x1 = (Pi_t @ x0) % MOD

    A1 = np.kron(np.identity(n1, dtype=object), Wn2) % MOD
    B1 = x1 % MOD
    C1 = (A1 @ B1) % MOD

    # Inject a random fault into C1
    i = rng.randrange(N); delta = rng.randrange(1, MOD)
    C1_fault = C1.copy()
    C1_fault[i, 0] = (C1_fault[i, 0] + delta) % MOD

    lhs1_ok, rhs1_ok, ok1_ok = protection_check(A1, B1, C1, MOD)
    lhs1_bad, rhs1_bad, ok1_bad = protection_check(A1, B1, C1_fault, MOD)

    print(f"[stage1 no-fault] ok={ok1_ok}, lhs={lhs1_ok}, rhs={rhs1_ok}")
    print(f"[stage1 fault at row {i}, +{delta}] ok={ok1_bad}, lhs={lhs1_bad}, rhs={rhs1_bad}")

if __name__ == "__main__":
    # Quick run
    demo(16, seed=42)
