# Optimize fold_mod_2k_plus_1 for fixed word size by limiting to 64 bits.
import random
from math import ceil, log2
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
# from caas_jupyter_tools import display_dataframe_to_user

rng = random.Random(8)

def choose_modulus(q_bits=30):
    q = (1 << q_bits) - 59
    K = ceil(log2(q))
    mu = (1 << (2*K)) // q
    return q, K, mu

def barrett_reduce_vector(t_vec, q, K, mu):
    s = ((t_vec * mu) >> (2*K)).astype(object)
    c = (t_vec - s*q).astype(object)
    c2 = c.copy()
    mask = c2 >= q
    c2[mask] = c2[mask] - q
    return s, c2

def fold_mod_2k_plus_1_64(x, k):
    M = (1 << k) + 1
    mask = (1 << k) - 1
    acc = 0
    sign = 1
    # limit to first 64 bits
    for i in range((64 + k -1)//k):
        seg = (x >> (i*k)) & mask
        acc = (acc + sign * seg) % M
        sign *= -1
    return acc % M

def fold_sum_vector(c_vec, k):
    M = (1 << k) + 1
    total = 0
    for v in c_vec:
        total = (total + fold_mod_2k_plus_1_64(int(v), k)) % M
    return total

def apply_bitflip(val, qbits, flips, rng):
    bits = rng.sample(range(qbits), flips)
    for b in bits:
        val ^= (1 << b)
    return val

def apply_mof(val, qbits, rng):
    mask = (1 << qbits) - 1
    new_low = rng.randrange(0, 1<<qbits)
    return (val & ~mask) | new_low

def inject_fault_s_or_c(t_i, s_i, c_i, q, qbits, mode, rng):
    target = rng.choice(['s','c'])
    if target == 's':
        if mode == 'SBF':
            s_fault = apply_bitflip(int(s_i), qbits, 1, rng)
        elif mode == 'DBF':
            s_fault = apply_bitflip(int(s_i), qbits, 2, rng)
        elif mode == 'MOF':
            s_fault = apply_mof(int(s_i), qbits, rng)
        c_fault = int(t_i) - s_fault * q
    else:
        c_fault = int(c_i)
        if mode == 'SBF':
            c_fault = apply_bitflip(c_fault, qbits, 1, rng)
        elif mode == 'DBF':
            c_fault = apply_bitflip(c_fault, qbits, 2, rng)
        elif mode == 'MOF':
            c_fault = apply_mof(c_fault, qbits, rng)
    if c_fault >= q:
        c_fault -= q
    detected_range = not (0 <= c_fault < q)
    return c_fault, detected_range

def run_reduction_experiment(n=8192, q_bits=30, qbits_flip=30,
                             k_list=tuple(range(4,34,2)), trials=1000000,
                             modes=('SBF','DBF','MOF')):
    q, K, mu = choose_modulus(q_bits)
    t_vec = np.array([rng.getrandbits(2*K) & ((1<<64)-1) for _ in range(n)], dtype=object)
    s_true, c_true = barrett_reduce_vector(t_vec, q, K, mu)
    base_fold_sum = {k: fold_sum_vector(c_true, k) for k in k_list}
    rows = []
    for mode in modes:
        for k in k_list:
            miss_range = 0
            miss_inter = 0
            M = (1<<k)+1
            base = base_fold_sum[k]
            for _ in tqdm(range(trials)):
                idx = rng.randrange(n)
                c_fault, detected_range = inject_fault_s_or_c(t_vec[idx], s_true[idx], c_true[idx], q, qbits_flip, mode, rng)
                before = fold_mod_2k_plus_1_64(int(c_true[idx]), k)
                after  = fold_mod_2k_plus_1_64(int(c_fault), k)
                faulty_sum = (base - before + after) % M
                detected_inter = (faulty_sum != base)
                if not detected_range:
                    miss_range += 1
                if not detected_inter:
                    miss_inter += 1
            rows.append({'strategy':'RangeCheck','mode':mode,'k':k,'miss_rate':miss_range/trials})
            rows.append({'strategy':'InterElem','mode':mode,'k':k,'miss_rate':miss_inter/trials})
    return pd.DataFrame(rows), {'q':q}

df_red, meta = run_reduction_experiment()
# display_dataframe_to_user("Reduction protection miss rate (k=2..10, trials=1000)", df_red)

def plot_strategy(df, strategy, path):
    plt.figure()
    print(df)
    df.to_csv("{}.csv".format(strategy))
    sub = df[df['strategy']==strategy].sort_values('k')
    for mode in sorted(sub['mode'].unique()):
        d = sub[sub['mode']==mode]
        plt.plot(d['k'].values, d['miss_rate'].values, marker='o', label=mode)
    plt.xlabel('fold width k')
    plt.ylabel('miss rate')
    plt.title(f'{strategy} vs fold width (mod 2^k+1)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)

plot_strategy(df_red, 'RangeCheck', 'reduction_rangecheck_vs_k.png')
plot_strategy(df_red, 'InterElem',  'reduction_inter_vs_k.png')

