# internal_rangecheck_injection.py
# Internal fault injection only: 50/50 at t*mu or s*q.
# Injection mode: flip 1 bit OR flip 2 bits.
# Checks: range-window before/after conditional subtract.

import argparse, random

BITWIDTH_PRIME = 37
VECTOR_LEN = 8192

# ---- 64-bit deterministic Miller–Rabin ----
def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    for p in [2,3,5,7,11,13,17,19,23,29,31,37]:
        if n % p == 0: return n == p
    d, r = n - 1, 0
    while d % 2 == 0: d //= 2; r += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0: continue
        x = pow(a, d, n)
        if x in (1, n - 1): continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1: break
        else:
            return False
    return True

def next_prime_ge(n: int) -> int:
    if n <= 2: return 2
    if n % 2 == 0: n += 1
    while not is_probable_prime(n): n += 2
    return n

Q = next_prime_ge(1 << (BITWIDTH_PRIME - 1))

# ---- Barrett params ----
def make_barrett_ctx(q: int):
    K = (q - 1).bit_length()        # ceil(log2 q)
    mu = (1 << (2 * K)) // q        # floor(2^(2K)/q)
    return K, mu

# ---- bit-flip helper ----
def flip_bits(x: int, rng: random.Random, k: int) -> int:
    """Flip k distinct random bit positions in x (allow growing width by 1)."""
    bl = max(1, x.bit_length() + 1)
    bits = rng.sample(range(bl), k if bl >= k else 1)
    for b in bits:
        x ^= (1 << b)
    return x

# ---- single reduction with internal injection + range window checks ----
def barrett_reduce_internal_fault(ti: int, q: int, K: int, mu: int,
                                  fault_prob: float, inj_bits: int,
                                  rng: random.Random) -> tuple[int, bool]:
    """
    Per element:
      - With probability fault_prob, inject ONCE at 50/50 either after mul1 (t*mu) or after mul2 (s*q).
      - Injection flips 'inj_bits' random bit(s) in the chosen intermediate.
      - Range check: pre-sub in [0, 2q), then conditional subtract once, then post-sub in [0, q).
    """
    # mul1 = ti * mu
    mul1 = ti * mu

    injected = False
    choose_mul1 = rng.getrandbits(1)  # 0/1, 50/50

    if rng.random() < fault_prob and choose_mul1:
        mul1 = flip_bits(mul1, rng, inj_bits)
        injected = True

    # s = floor(mul1 / 2^(2K))
    si = mul1 >> (2 * K)

    # mul2 = si * q
    mul2 = si * q

    if (not injected) and (rng.random() < fault_prob):
        mul2 = flip_bits(mul2, rng, inj_bits)
        injected = True

    # ci before conditional subtract
    ci = ti - mul2

    ok = True
    if not (0 <= ci < 2 * q):    # pre-sub window
        ok = False

    if ci >= q:
        ci -= q

    if not (0 <= ci < q):        # post-sub window
        ok = False

    return ci, ok

# ---- runner ----
def run(trials: int, fault_prob: float, inj_bits: int, seed: int, n: int):
    rng = random.Random(seed)
    K, mu = make_barrett_ctx(Q)
    print(f"q={Q} (bits={Q.bit_length()}), N={n}")
    print(f"Internal faults only | 50/50 at t*mu vs s*q | fault_prob={fault_prob} | inj_bits={inj_bits}")

    det_range, undetected = 0, 0

    for _ in range(trials):
        ok_all = True
        for _i in range(n):
            a = rng.randrange(Q); b = rng.randrange(Q)
            ti = a * b
            _, ok = barrett_reduce_internal_fault(ti, Q, K, mu, fault_prob, inj_bits, rng)
            if not ok:
                ok_all = False  # at least one element flagged by range check
        if ok_all:
            undetected += 1
        else:
            det_range += 1

    miss_rate = undetected / max(1, trials)
    print(f"Trials={trials} | Det(Range)={det_range} | Undetected={undetected} | MissRate={miss_rate:.6f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Internal range-check with 50/50 injection at t*mu / s*q")
    ap.add_argument("--trials", type=int, default=200, help="Number of vector trials")
    ap.add_argument("--fault-prob", type=float, default=0.01, help="Per-element injection probability")
    ap.add_argument("--inj-bits", type=int, default=1, choices=[1,2], help="Flip 1 or 2 bits")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vec-len", type=int, default=VECTOR_LEN)
    args = ap.parse_args()
    run(args.trials, args.fault_prob, args.inj_bits, args.seed, args.vec_len)
