import random
import math
from sympy import randprime
from sympy.ntheory.modular import crt
import pandas as pd

# 1. 指定位宽列表
mods_bit_widths = [40] * 7   # 7 个 40-bit 小模
rems_group_count = 3         # 最终分成 3 组

# 2. 生成对应比特宽度的随机素数模 (distinct)
primes = []
seen = set()
for b in mods_bit_widths:
    while True:
        p = randprime(2**(b-1), 2**b - 1)
        if p not in seen:
            seen.add(p)
            primes.append(p)
            break

# 3. 按组划分小模

def split_groups(lst, n_groups):
    L = len(lst)
    base = L // n_groups
    extra = L % n_groups
    groups = []
    idx = 0
    for i in range(n_groups):
        size = base + (1 if i < extra else 0)
        groups.append(lst[idx:idx+size])
        idx += size
    return groups

prime_groups = split_groups(primes, rems_group_count)

# 4. 计算每组合并后的大模 Q_k
big_mods = [math.prod(g) for g in prime_groups]

# 5. 为每个大模生成随机残余 rem_k
big_rems = [random.getrandbits(40) % Qk for Qk in big_mods]

# 6. 使用 CRT 重建 x
x, M = crt(big_mods, big_rems)

# 7. 验证：检查 x % Q_k == rem_k
for i, (Qk, rk) in enumerate(zip(big_mods, big_rems)):
    assert x % Qk == rk, f"Group {i} CRT mismatch: x%Qk={x%Qk}, rk={rk}"
    print(f"Group {i} check passed: x % Qk == {rk}")

# 8. 打印结果
print(f"Reconstructed x = {x}")
print(f"Combined modulus M = {M}")

# 9. 展示表格

df_small = pd.DataFrame({
    'prime_modulus': primes,
    'bit_width': mods_bit_widths,
    'group_id': sum([[i]*len(g) for i,g in enumerate(prime_groups)], [])
})
df_big = pd.DataFrame({
    'group': list(range(rems_group_count)),
    'big_modulus': big_mods,
    'big_remainder': big_rems
})

print("\nSmall primes and their groups:")
print(df_small)
print("\nGroup moduli and remainders:")
print(df_big)
