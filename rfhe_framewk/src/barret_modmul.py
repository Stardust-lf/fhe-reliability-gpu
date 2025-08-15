import random
from typing import List, Tuple
from tqdm import tqdm

# ==================== Hyperparameters ====================
BITWIDTH_PRIME     = 37      # prime modulus bitwidth
VECTOR_LEN         = 8192    # vector length
TRIALS             = 100000    # Monte Carlo trials per (fold_width, scheme)
FOLD_WIDTHS        = list(range(2, 34, 2))
SCHEMES            = [       # (USE_T_CHECK, USE_SN_CHECK, USE_FINAL_CHECK)
    (True,  False, False),
    (False, True,  False),
    (False, False, True),
]
# Fault injection in two dimensions
INJECT_ELEM_COUNT  = 1       # how many elements to corrupt in a trial
BITFLIPS_PER_ELEM  = 1       # how many distinct bits to flip in each corrupted element
# ==========================================================

# ---------- primality and Barrett constants ----------
def is_prime(n, k=5):
    """Miller-Rabin primality test."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def gen_prime(bitwidth: int) -> int:
    """Generate a random prime with given bitwidth."""
    while True:
        p = random.getrandbits(bitwidth) | (1 << (bitwidth - 1)) | 1
        if is_prime(p):
            return p

def compute_mu(n: int) -> Tuple[int, int]:
    """Return (mu, k) where k = bitlen(n), mu = floor(2^(2k)/n)."""
    k = n.bit_length()
    mu = (1 << (2 * k)) // n
    return mu, k

# ---------- Folding ECC ----------
def fold_mod(x: int, width: int) -> int:
    """Fold x into width-bit chunks modulo (2^width - 1)."""
    if width <= 0:
        raise ValueError("width must be positive")
    M = (1 << width) - 1
    s = 0
    while x:
        s += x & M
        x >>= width
    return s % M

# ---------- Fault injection helpers ----------
def flip_bits(value: int, bit_positions: List[int]) -> int:
    """Flip bits of value at given positions."""
    out = value
    for pos in bit_positions:
        out ^= (1 << pos)
    return out

def inject_multi(values: List[int], elem_count: int, bits_per_elem: int):
    """
    Corrupt 'elem_count' distinct elements. For each chosen element,
    flip 'bits_per_elem' distinct bit positions.
    Returns: new_values, flip_infos (list of tuples per element)
    tuple format: (idx, bit_positions, before, after)
    """
    if not values or elem_count <= 0:
        return values, []

    n = len(values)
    elem_count = min(elem_count, n)
    indices = random.sample(range(n), elem_count)

    new_vals = values.copy()
    infos = []
    for idx in indices:
        v = new_vals[idx]
        # handle zero value: define at least 1-bit space
        bitlen = max(1, v.bit_length())
        k = min(bits_per_elem, bitlen)
        # choose distinct bit positions in [0, bitlen-1]
        positions = random.sample(range(bitlen), k) if k > 0 else []
        before = v
        after = flip_bits(v, positions)
        new_vals[idx] = after
        infos.append((idx, positions, before, after))

    return new_vals, infos

# ---------- Vector Barrett (correct path) ----------
def barrett_reduce_vector_correct(X: List[int], n: int, mu: int, k: int) -> List[int]:
    """Correct Barrett reduction for a vector of x_i."""
    out = []
    for x in X:
        t = x * mu
        s = t >> (2 * k)
        c = x - s * n
        if c < 0:
            c += n
        elif c >= n:
            c -= n
        out.append(c)
    return out

# ---------- Vector Barrett with multi-element faults + detectors ----------
def barrett_reduce_vector_faulty_with_checks(
    X: List[int], n: int, mu: int, k: int, width: int,
    use_t_check: bool, use_sn_check: bool, use_final_check: bool,
    elem_count: int, bits_per_elem: int
):
    """
    One fault campaign per trial with two dimensions:
      - elem_count: how many elements are corrupted
      - bits_per_elem: how many distinct bits flipped per corrupted element
    Injection site chosen 50/50 between T-stage and SN-stage.
    Detectors:
      - T ECC: compare fold(sum t_i) before vs after
      - SN ECC: compare fold(sum sn_i) before vs after
      - Final range: per element check 0 <= c_i < 2n before correction
    """
    # T-stage
    t_list = [x * mu for x in X]
    ecc_t_before = fold_mod(sum(t_list), width) if use_t_check else None

    # s_i and SN
    s_list = [t >> (2 * k) for t in t_list]
    sn_list = [s * n for s in s_list]
    ecc_sn_before = fold_mod(sum(sn_list), width) if use_sn_check else None

    inject_stage = random.choice(["T", "SN"])
    meta = {"stage": inject_stage, "flips": []}

    # apply faults
    if inject_stage == "T":
        t_list_fault, infos = inject_multi(t_list, elem_count, bits_per_elem)
        meta["flips"] = infos
        t_list = t_list_fault
        # refresh downstream
        s_list = [t >> (2 * k) for t in t_list]
        sn_list = [s * n for s in s_list]
    else:
        sn_list_fault, infos = inject_multi(sn_list, elem_count, bits_per_elem)
        meta["flips"] = infos
        sn_list = sn_list_fault

    # detections
    detect_t = False
    detect_sn = False
    if use_t_check and inject_stage == "T":
        ecc_t_after = fold_mod(sum(t_list), width)
        detect_t = (ecc_t_before != ecc_t_after)
    if use_sn_check:
        ecc_sn_after = fold_mod(sum(sn_list), width)
        detect_sn = (ecc_sn_before != ecc_sn_after)

    # final stage
    c_list = []
    final_flags = []
    for x, sn in zip(X, sn_list):
        c = x - sn
        flag = not (0 <= c < 2 * n) if use_final_check else False
        if c < 0:
            c += n
        elif c >= n:
            c -= n
        c_list.append(c)
        final_flags.append(flag)

    detected_any = detect_t or detect_sn or (any(final_flags) if use_final_check else False)
    return c_list, detected_any, meta

# ---------- Monte Carlo ----------
def run_experiment(fold_width: int, scheme: Tuple[bool, bool, bool]):
    use_t, use_sn, use_final = scheme
    p = gen_prime(BITWIDTH_PRIME)
    mu, k = compute_mu(p)

    TP = FN = TN = FP = 0
    desc = (f"fw={fold_width}, scheme={int(use_t)}{int(use_sn)}{int(use_final)}"
            f", elems={INJECT_ELEM_COUNT}, bits={BITFLIPS_PER_ELEM}")

    for _ in tqdm(range(TRIALS), desc=desc, leave=False):
        A = [random.randrange(1, p) for _ in range(VECTOR_LEN)]
        B = [random.randrange(1, p) for _ in range(VECTOR_LEN)]
        X = [a * b for a, b in zip(A, B)]

        out_ok = barrett_reduce_vector_correct(X, p, mu, k)
        out_fault, detected_any, meta = barrett_reduce_vector_faulty_with_checks(
            X, p, mu, k, fold_width, use_t, use_sn, use_final,
            INJECT_ELEM_COUNT, BITFLIPS_PER_ELEM
        )

        harmful = (out_fault != out_ok)
        if harmful:
            if detected_any: TP += 1
            else:            FN += 1
        else:
            if detected_any: FP += 1
            else:            TN += 1

    total_harmful = TP + FN
    detection_rate = (TP / total_harmful) if total_harmful > 0 else 1.0
    return detection_rate, TP, FP, TN, FN

def main():
    print("FOLD_WIDTH, USE_T, USE_SN, USE_FINAL, DetectionRate, TP, FP, TN, FN")
    for fw in FOLD_WIDTHS:
        for scheme in SCHEMES:
            rate, TP, FP, TN, FN = run_experiment(fw, scheme)
            print(f"{fw}, {int(scheme[0])}, {int(scheme[1])}, {int(scheme[2])}, "
                  f"{rate:.4f}, {TP}, {FP}, {TN}, {FN}")

if __name__ == "__main__":
    main()
