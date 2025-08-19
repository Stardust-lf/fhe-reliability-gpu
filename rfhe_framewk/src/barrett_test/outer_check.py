# 优化：预先把 base 校验和按 k 增量维护，减少重复计算；并限制 n=4096, trials=1000 仍可运行。
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
# from caas_jupyter_tools import display_dataframe_to_user

rng = random.Random(4)

def gen_c(n, word_bits=64):
    return [rng.getrandbits(word_bits) for _ in range(n)]

def inject_and_delta(c_true, qbits, mode):
    n = len(c_true)
    qmask = (1 << qbits) - 1
    deltas = {}
    changed = defaultdict(list)

    def flip_from_old(old, bit):
        bitmask = (1 << bit)
        return bitmask if (old & bitmask) == 0 else -bitmask

    indices = list(range(n))
    if mode == 'SBF':
        i = rng.choice(indices); b = rng.randrange(0, qbits); changed[i].append(b)
    elif mode == 'DBF':
        i = rng.choice(indices); b1, b2 = rng.sample(range(qbits), 2); changed[i] += [b1,b2]
    elif mode == 'SBF+SBF':
        i1,i2 = rng.sample(indices,2); changed[i1].append(rng.randrange(0,qbits)); changed[i2].append(rng.randrange(0,qbits))
    elif mode == 'SBF+DBF':
        i1,i2 = rng.sample(indices,2); changed[i1].append(rng.randrange(0,qbits)); b2,b3 = rng.sample(range(qbits),2); changed[i2]+=[b2,b3]
    elif mode == 'MOF1':
        i = rng.choice(indices); changed[i].append('RAND')
    elif mode == 'MOF2':
        i1,i2 = rng.sample(indices,2); changed[i1].append('RAND'); changed[i2].append('RAND')

    elif mode == 'MOF+SBF':
        i1, i2 = rng.sample(indices, 2)
        changed[i1].append('RAND')
        changed[i2].append(rng.randrange(0, qbits))

    elif mode == 'MOF+DBF':
        i1, i2 = rng.sample(indices, 2)
        changed[i1].append('RAND')
        b2, b3 = rng.sample(range(qbits), 2)
        changed[i2] += [b2, b3]

    for idx, ops in changed.items():
        old = c_true[idx]
        delta = 0
        for op in ops:
            if op == 'RAND':
                old_low = old & qmask
                new_low = rng.randrange(0, 1<<qbits)
                delta += (new_low - old_low)
                old = (old & ~qmask) | new_low
            else:
                delta += flip_from_old(old, op)
                old ^= (1 << op)
        deltas[idx] = delta
    return deltas

def fold_mod_2k_plus_1(x, k):
    M = (1 << k) + 1
    mask = (1 << k) - 1
    # 只需处理到 64 位；最多 ceil(64/k) 段
    acc = 0
    sign = 1
    segs = (64 + k - 1)//k
    for i in range(segs):
        seg = (x >> (i*k)) & mask
        acc += sign * seg
        acc %= M
        sign *= -1
    return acc % M

def precompute_base_checksums(c_true, ks):
    # 返回：base[k] = sum(fold(c_i, k)) mod M_k
    bases = {}
    for k in ks:
        M = (1<<k)+1
        s = 0
        for v in c_true:
            s = (s + fold_mod_2k_plus_1(v, k)) % M
        bases[k] = s
    return bases

def run_experiment(n=4, qbits=16, folds=tuple(range(2,34,2)), trials=10000000,
                   modes=('SBF','DBF','SBF+SBF','SBF+DBF','MOF1','MOF2', 'MOF+SBF', 'MOF+DBF'),
                   # modes=('MOF+SBF', 'MOF+DBF'),
                   use_intra=True, use_inter=True):
    c_true = gen_c(n)
    base_map = precompute_base_checksums(c_true, folds)
    rows = []
    for k in folds:
        M = (1<<k)+1
        for mode in modes:
            miss_intra = 0
            miss_inter = 0
            for _ in tqdm(range(trials)):
                deltas = inject_and_delta(c_true, qbits, mode)
                # intra：逐元素比较
                d_intra = True
                if use_intra:
                    d_intra = False
                    for idx, d in deltas.items():
                        before = fold_mod_2k_plus_1(c_true[idx], k)
                        after  = fold_mod_2k_plus_1((c_true[idx] + d) & ((1<<64)-1), k)
                        if before != after:
                            d_intra = True
                            break
                # inter：总校验和比较（利用预计算 base）
                d_inter = True
                if use_inter:
                    delta_sum = 0
                    for d in deltas.values():
                        delta_sum = (delta_sum + fold_mod_2k_plus_1(d & ((1<<64)-1), k)) % M
                    faulty = (base_map[k] + delta_sum) % M
                    d_inter = (faulty != base_map[k])
                if not d_intra: miss_intra += 1
                if not d_inter: miss_inter += 1
            rows.append({'strategy':'intra','fold':k,'mode':mode,'miss_rate':miss_intra/trials})
            rows.append({'strategy':'inter','fold':k,'mode':mode,'miss_rate':miss_inter/trials})
    return pd.DataFrame(rows)

df_k = run_experiment()
# display_dataframe_to_user("Miss rate with mod (2^k+1) folding (k=2..10)", df_k)

# 绘图
def plot_by_strategy(df, strategy, path):
    import matplotlib.pyplot as plt
    plt.figure()
    sub = df[df['strategy']==strategy]
    for mode in sorted(sub['mode'].unique()):
        d = sub[sub['mode']==mode].sort_values('fold')
        plt.plot(d['fold'].values, d['miss_rate'].values, marker='o', label=mode)
    plt.xlabel('fold width k')
    plt.ylabel('miss rate')
    plt.title(f'{strategy} (mod 2^k+1) miss rate vs k')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)

plot_by_strategy(df_k, 'intra', 'outer_intra.png')
plot_by_strategy(df_k, 'inter', 'outer_inter.png')

csv_path = "outer_check.csv"
df_k.to_csv(csv_path, index=False)
print(csv_path)
# print("/mnt/data/intra_mod2kp1.png")
# print("/mnt/data/inter_mod2kp1.png")
