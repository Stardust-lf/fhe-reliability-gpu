import random
from typing import List, Tuple
from tqdm import tqdm

# ==================== Hyperparameters ====================
BITWIDTH_PRIME   = 37
VECTOR_LEN       = 8192
TRIALS           = 1000
BITFLIP_COUNT    = 2
# ==========================================================

def is_prime(n, k=5):
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
    while True:
        p = random.getrandbits(bitwidth) | (1 << (bitwidth - 1)) | 1
        if is_prime(p):
            return p

def compute_montgomery_constants(p: int) -> Tuple[int, int, int]:
    k = p.bit_length()
    R = 1 << k
    inv = pow(p, -1, R)
    m_prime = (-inv) & (R - 1)
    return R, k, m_prime

def fold_mod(x: int, width: int) -> int:
    M = (1 << width) - 1
    mask = M
    s = 0
    while x:
        s += x & mask
        x >>= width
    return s % M

def inject_bitflip_list(values: List[int], flip_count: int):
    """Flip bits in a random element, return new list and flip info."""
    if not values:
        return values, None
    idx = random.randrange(len(values))
    val = values[idx]
    bitlen = max(1, val.bit_length())
    flip_count = min(flip_count, bitlen)
    bit_positions = random.sample(range(bitlen), flip_count)
    val_flipped = val
    for bitpos in bit_positions:
        val_flipped ^= (1 << bitpos)
    new_values = values.copy()
    new_values[idx] = val_flipped
    return new_values, (idx, bit_positions, val, val_flipped)

def montgomery_reduce_list(
    T_list: List[int], p: int, R: int, k: int, m_prime: int
) -> List[int]:
    maskR = R - 1
    lowT_list = [t & maskR for t in T_list]
    m_prime_list = [m_prime] * len(T_list)
    prod_m_list = [lt * mp for lt, mp in zip(lowT_list, m_prime_list)]
    m_list = [(val & maskR) for val in prod_m_list]
    p_list = [p] * len(T_list)
    prod_mp_list = [m * p for m, p in zip(m_list, p_list)]
    out_list = []
    for t, mp in zip(T_list, prod_mp_list):
        T_val = t + mp
        u2 = T_val >> k
        u = u2 - p if u2 >= p else u2
        out_list.append(u)
    return out_list

def montgomery_reduce_list_ecc_inject_detect(
    T_list, p, R, k, m_prime, width, flip_count,
    use_m_check, use_mp_check, use_final_check
):
    maskR = R - 1
    lowT_list = [t & maskR for t in T_list]
    m_prime_list = [m_prime] * len(T_list)
    prod_m_list = [lt * mp for lt, mp in zip(lowT_list, m_prime_list)]
    ecc_before_m = fold_mod(sum(prod_m_list), width) if use_m_check else None

    m_list = [(val & maskR) for val in prod_m_list]
    p_list = [p] * len(T_list)
    prod_mp_list = [m * p for m, p in zip(m_list, p_list)]
    ecc_before_mp = fold_mod(sum(prod_mp_list), width) if use_mp_check else None

    inject_stage = random.choice(["m", "mp"])
    flip_info = None

    if inject_stage == "m":
        prod_m_list_fault, flip_info = inject_bitflip_list(prod_m_list, flip_count)
        m_list = [(val & maskR) for val in prod_m_list_fault]
    else:
        prod_mp_list = [m * p for m, p in zip(m_list, p_list)]
        prod_mp_list_fault, flip_info = inject_bitflip_list(prod_mp_list, flip_count)
        prod_mp_list = prod_mp_list_fault

    if inject_stage == "m":
        prod_mp_list = [m * p for m, p in zip(m_list, p_list)]

    detect_m = False
    detect_mp = False
    if use_m_check and inject_stage == "m":
        ecc_after_m = fold_mod(sum(prod_m_list_fault), width)
        detect_m = (ecc_before_m != ecc_after_m)
    if use_mp_check:
        ecc_after_mp = fold_mod(sum(prod_mp_list), width)
        detect_mp = (ecc_before_mp != ecc_after_mp)

    detect_final_flags = []
    if use_final_check:
        for t, mp in zip(T_list, prod_mp_list):
            T_val = t + mp
            detect_final_flags.append((T_val & maskR) != 0)

    # Faulty output
    out_list_fault = []
    for t, mp in zip(T_list, prod_mp_list):
        T_val = t + mp
        u2 = T_val >> k
        u = u2 - p if u2 >= p else u2
        out_list_fault.append(u)

    return detect_m, detect_mp, detect_final_flags, flip_info, out_list_fault

def run_experiment(fold_width, use_m, use_mp, use_final):
    TP_total = FN_total = 0
    for _ in range(TRIALS):
        p = gen_prime(BITWIDTH_PRIME)
        R, k, m_prime = compute_montgomery_constants(p)
        A = [random.randrange(1, p) for _ in range(VECTOR_LEN)]
        B = [random.randrange(1, p) for _ in range(VECTOR_LEN)]
        T_list = [a * b for a, b in zip(A, B)]

        # correct output
        out_correct = montgomery_reduce_list(T_list, p, R, k, m_prime)

        detect_m, detect_mp, detect_final_flags, flip_info, out_fault = montgomery_reduce_list_ecc_inject_detect(
            T_list, p, R, k, m_prime, fold_width, BITFLIP_COUNT,
            use_m, use_mp, use_final
        )

        faulty_has_effect = (out_fault != out_correct)
        detected_any = detect_m or detect_mp or (use_final and any(detect_final_flags))

        if faulty_has_effect and not detected_any:
            FN_total += 1
            stage, bits_idx, before, after = flip_info[0], flip_info[1], flip_info[2], flip_info[3]
            # print(f"[MISS] stage={stage}, idx={bits_idx}, before={before}, after={after}")
        elif faulty_has_effect and detected_any:
            TP_total += 1
        # harmless fault is ignored

    return TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 1.0

def main():
    # fold_widths = list(range(12, 26, 2))
    fold_widths = list(range(26, 34, 2))
    schemes = [
        (True, False, True),
        (False, True, True),
        (True, True, True)
    ]
    print("FOLD_WIDTH, USE_M, USE_MP, DetectionRate")
    for fw in fold_widths:
        for scheme in schemes:
            rate = run_experiment(fw, *scheme)
            print(f"{fw}, {int(scheme[0])}, {int(scheme[1])}, {rate:.4f}")

if __name__ == "__main__":
    main()
