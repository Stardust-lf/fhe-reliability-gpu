# checksum_min.py
# Minimal Python version with three protections and optional reduce-path faults.
# All comments in English. Dialog in Chinese.

import argparse, random

# ---------- Hyperparameters ----------
BITWIDTH_PRIME = 37
VECTOR_LEN     = 8192
FOLD_WIDTHS    = list(range(4, 11))  # s in {4..10}

# ---------- Small 64-bit prime builder ----------
def is_probable_prime(n: int) -> bool:
    if n < 2: return False
    small = [2,3,5,7,11,13,17,19,23,29,31,37]
    for p in small:
        if n % p == 0: return n == p
    # n-1 = d*2^r
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2; r += 1
    for a in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if a % n == 0: 
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1: 
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1: 
                break
        else:
            return False
    return True

def next_prime_ge(n: int) -> int:
    if n <= 2: return 2
    if n % 2 == 0: n += 1
    while not is_probable_prime(n):
        n += 2
    return n

Q = next_prime_ge(1 << (BITWIDTH_PRIME - 1))  # fixed 37-bit prime

# ---------- Folding checksum x mod (2^s+1) ----------
def residue_mod_2s1(x: int, s: int) -> int:
    base = 1 << s
    mod  = base + 1
    acc, sign = 0, 1
    while x:
        acc += sign * (x & (base - 1))
        sign = -sign
        x >>= s
    return acc % mod

# ---------- Fault models BEFORE reduction ----------
def inject_faults(c, model, rng):
    n = len(c); out = c[:]
    if model == 1:
        i = rng.randrange(n)
        out[i] = rng.getrandbits(max(1, out[i].bit_length()+1))
    elif model == 2:
        i, j = rng.sample(range(n), 2 if n > 1 else 1)
        out[i] = rng.getrandbits(max(1, out[i].bit_length()+1))
        if n > 1:
            out[j] = rng.getrandbits(max(1, out[j].bit_length()+1))
    elif model == 3:
        i = rng.randrange(n)
        x = out[i]
        bl = max(1, x.bit_length()+1)
        b1, b2 = rng.sample(range(bl), 2 if bl > 1 else 1)
        x ^= (1<<b1)
        if bl > 1: x ^= (1<<b2)
        out[i] = x
    elif model == 4:
        idxs = rng.sample(range(n), 2 if n > 1 else 1)
        for i in idxs:
            x = out[i]; bl = max(1, x.bit_length()+1); b = rng.randrange(bl)
            out[i] = x ^ (1<<b)
    return out

# ---------- Barrett with range-window checks + optional reduce-path faults ----------
def barrett_reduce_ci(ti: int, q: int, K: int, mu: int, use_range: bool,
                      fault_on_reduce: bool, fault_prob: float, rng: random.Random) -> tuple[int, bool]:
    # mul1 = ti * mu
    mul1 = ti * mu
    # 50/50 decide which multiply to corrupt if firing
    choose_mul1 = rng.getrandbits(1)
    if fault_on_reduce and rng.random() < fault_prob and choose_mul1:
        # flip one random bit in mul1
        b = rng.randrange(max(1, mul1.bit_length()+1))
        mul1 ^= (1 << b)
    # s_i = floor(mul1 / 2^(2K))
    si = mul1 >> (2*K)
    # mul2 = si * q
    mul2 = si * q
    if fault_on_reduce and rng.random() < fault_prob and not choose_mul1:
        b = rng.randrange(max(1, mul2.bit_length()+1))
        mul2 ^= (1 << b)
    # ci before conditional subtract
    ci = ti - mul2
    ok = True
    if use_range and not (0 <= ci < 2*q):
        ok = False
    # conditional subtract once
    if ci >= q: 
        ci -= q
    if use_range and not (0 <= ci < q):
        ok = False
    return ci, ok

def make_barrett_ctx(q: int) -> tuple[int,int]:
    K = (q-1).bit_length()  # ceil(log2 q)
    mu = (1 << (2*K)) // q  # floor(2^(2K)/q)
    return K, mu

# ---------- Runner ----------
def run(trials_per_s: int, use_intra: bool, use_range: bool, use_sum: bool,
        seed: int, fault_on_reduce: bool, fault_prob: float):
    rng = random.Random(seed)
    K, mu = make_barrett_ctx(Q)
    print(f"q={Q} (bits={Q.bit_length()}), N={VECTOR_LEN}")
    prot = " ".join([x for x,flag in (("Intra",use_intra),("Range",use_range),("Sum",use_sum)) if flag]) or "None"

    for s in FOLD_WIDTHS:
        for model in (1,2,3,4):
            det_intra_only = det_sum_only = det_both = undetected = 0
            for _ in range(trials_per_s):
                a = [rng.randrange(Q) for _ in range(VECTOR_LEN)]
                b = [rng.randrange(Q) for _ in range(VECTOR_LEN)]
                c_true = [ai*bi for ai,bi in zip(a,b)]
                t_tot_ref = sum(c_true)

                c_faulty = inject_faults(c_true, model, rng)

                intra_ok = True
                if use_intra:
                    qs = (1<<s)+1
                    a_res = [residue_mod_2s1(x, s) for x in a]
                    b_res = [residue_mod_2s1(x, s) for x in b]
                    for i in range(VECTOR_LEN):
                        if residue_mod_2s1(c_faulty[i], s) != (a_res[i]*b_res[i]) % qs:
                            intra_ok = False; break

                inter_ok = True
                if use_sum or use_range:
                    sum_red = 0
                    for x in c_faulty:
                        ci, ok = barrett_reduce_ci(x, Q, K, mu, use_range, fault_on_reduce, fault_prob, rng)
                        if not ok:
                            inter_ok = False; break
                        sum_red += ci
                    if inter_ok and use_sum:
                        inter_ok = (sum_red % Q) == (t_tot_ref % Q)

                if (not intra_ok) and (not inter_ok): det_both += 1
                elif (not intra_ok) and inter_ok:      det_intra_only += 1
                elif intra_ok and (not inter_ok):      det_sum_only += 1
                else:                                   undetected += 1

            miss = undetected / max(1,trials_per_s)
            desc = {1:"Type1: randomize one",2:"Type2: randomize two",
                    3:"Type3: flip two bits (one elem)",4:"Type4: flip one bit (two elems)"}[model]
            extra = f" | ReducePathFaultProb={fault_prob}" if fault_on_reduce else ""
            print(f"Fold s={s} | Protection={{{prot}}} | Fault={desc} | Trials={trials_per_s}"
                  f" | MissRate={miss:.6f} | Det(IntraOnly)={det_intra_only} | Det(SumOnly)={det_sum_only}"
                  f" | Det(Both)={det_both} | Undetected={undetected}{extra}")

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Minimal checksum framework (Python)")
    ap.add_argument("--trials-per-s", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-intra", action="store_true")
    ap.add_argument("--no-range", action="store_true")
    ap.add_argument("--no-sum", action="store_true")
    ap.add_argument("--fault-on-reduce", action="store_true", help="Inject faults inside Barrett path")
    ap.add_argument("--fault-prob", type=float, default=0.0, help="Per-element prob for reduce-path fault")
    args = ap.parse_args()

    run(trials_per_s=args.trials_per_s,
        use_intra=not args.no_intra,
        use_range=not args.no_range,
        use_sum=not args.no_sum,
        seed=args.seed,
        fault_on_reduce=args.fault_on_reduce,
        fault_prob=args.fault_prob)
