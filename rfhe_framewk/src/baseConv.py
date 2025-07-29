from typing import List
import random
import math
from sympy import nextprime
import copy
from tqdm import tqdm
import numpy as np


def bConv(
    residue_arrays: List[List[int]], moduli: List[int], moduli_out: List[int]
) -> List[int]:
    m = len(moduli)
    N = len(residue_arrays[0])
    P = math.prod(moduli)

    hat_p = [P // p for p in moduli]
    inv_hat_p = [pow(hat_p[j], -1, moduli[j]) for j in range(m)]

    result = []
    for i in range(N):
        x = 0
        main_limb = []
        for j in range(m):
            sub_limb = []
            for q in moduli_out:
                r = int(residue_arrays[j][i])
                term = (r * hat_p[j] * inv_hat_p[j])
                sub_limb.append(term % q)
            main_limb.append(sub_limb)

        summed = []
        for k in range(len(moduli_out)):
            total = 0
            for j in range(m):
                total += main_limb[j][k]
            summed.append(total)
        result.append(summed)
    # print(np.array(result))
    return result

def bConv_ecc(
    residue_arrays: List[List[int]], moduli: List[int], moduli_out: List[int]
) -> List[int]:
    m = len(moduli)
    P = math.prod(moduli)

    for i in range(m):
        residue_arrays[i].append(sum(residue_arrays[i]) % moduli[i])

    N = len(residue_arrays[0])

    hat_p = [P // p for p in moduli]
    inv_hat_p = [pow(hat_p[j], -1, moduli[j]) for j in range(m)]

    result = []
    full_data = []
    for i in range(N):
        x = 0
        main_limb = []
        for j in range(m):
            sub_limb = []
            for q in moduli_out:
                r = int(residue_arrays[j][i])
                term = (r * hat_p[j] * inv_hat_p[j])
                sub_limb.append(term % q)
            # print(sum(sub_limb[:-1]), sub_limb[-1])
            main_limb.append(sub_limb)
        full_data.append(main_limb)

        summed = []
        for k in range(len(moduli_out)):
            total = 0
            for j in range(m):
                total += main_limb[j][k]
            summed.append(total)
        result.append(summed)
    # —— 在 bConv_ecc 中，累加前 N‑1 项时做模 q ——  
    sum_data = [[0]*len(moduli_out) for _ in range(m)]
    for item in full_data[:-1]:
        for j in range(m):
            for k, q in enumerate(moduli_out):
                # 每步都 mod q，保持在 [0..q-1]
                sum_data[j][k] = (sum_data[j][k] + item[j][k]) % q

    # ECC 项本身就是 mod q 后的
    ecc_item = full_data[-1]

    # 再比较
    for j in range(m):
        for k, q in enumerate(moduli_out):
            if sum_data[j][k] != ecc_item[j][k]:
                raise ValueError(
                    f"ECC mismatch at j={j}, k={k}: "
                    f"sum_mod={sum_data[j][k]}, ecc={ecc_item[j][k]}"
                )

    return result

def generate_crt_primes(limbs: int, bitwidth: int) -> List[int]:
    if bitwidth < 10:
        raise ValueError("bitwidth must be at least 10")
    primes = []
    seen = set()
    while len(primes) < limbs:
        cand = random.getrandbits(bitwidth) | (1 << (bitwidth - 1)) | 1
        p = int(nextprime(cand))
        if p.bit_length() == bitwidth and p not in seen:
            primes.append(p)
            seen.add(p)
    return primes


def generate_residues(moduli: List[int], poly_dim: int) -> List[List[int]]:
    residues = []
    for p in moduli:
        arr = [random.randrange(p) for _ in range(poly_dim)]
        residues.append(arr)
    return residues


import random
from typing import List, Optional, Tuple


def flip_n_bits_across_m_elements_2d_inplace(
    matrix: List[List[int]],
    m: int,
    n: int,
    bit_width: Optional[int] = None,
    max_bit: Optional[int] = None,
) -> None:
    """
    在二维整数矩阵中，随机选 m 个不同元素，然后在这 m 个元素的所有位位置里
    随机翻转总共 n 个位（直接在原矩阵上修改，无返回值）。

    Args:
        matrix:   二维列表，matrix[row][col] 是一个整数。
        m:        要选取的元素个数，必须 1 <= m <= 总元素数。
        n:        要翻转的总位数，必须 0 <= n <= m * 最大位宽。
        bit_width:
                  如果指定，则所有元素都按这个固定的位宽来考虑位位置 [0..bit_width-1]；
                  否则按元素自身的 bit_length()（至少 1）。
        max_bit:
                  如果指定，则只允许翻转位索引 <= max_bit 的位，
                  也就是最终可选 bitpos 范围会被截断到 [0..max_bit]。

    Raises:
        ValueError: 当 m 或 n 超出合理范围时。
    """
    # 扁平化所有元素的位置
    all_pos = [(r, c) for r in range(len(matrix)) for c in range(len(matrix[r]))]
    total = len(all_pos)
    if not (1 <= m <= total):
        raise ValueError(f"m must be between 1 and {total}, got {m}")

    # 随机选 m 个不同元素
    chosen = random.sample(all_pos, m)

    # 构造所有 (r, c, bitpos) 候选
    cand_bits: List[Tuple[int, int, int]] = []
    for r, c in chosen:
        val = matrix[r][c]
        # 确定位宽
        if bit_width is not None:
            w = bit_width
        else:
            w = val.bit_length() or 1
        # 如果 max_bit 限制了最高可翻转位，则截断
        if max_bit is not None:
            w = min(w, max_bit + 1)
        for b in range(w):
            cand_bits.append((r, c, b))

    if not (0 <= n <= len(cand_bits)):
        raise ValueError(f"n must be between 0 and {len(cand_bits)}, got {n}")

    # 随机选 n 个 (r, c, b) 并翻转
    for r, c, b in random.sample(cand_bits, n):
        matrix[r][c] ^= 1 << b


def group_multiply(lst, n):
    result = []
    for i in range(0, len(lst), n):
        group = lst[i : i + n]
        if len(group) < n:
            break
        product = 1
        for x in group:
            product *= x
        result.append(product)
    return result


if __name__ == "__main__":
    limbs, limbs_out, crt_bitwidth, poly_dim = 4, 2, 10, 1024
    ecc_pass = 0
    ecc_fail = 0
    crt_fail = 0
    flip_bits = 2
    flip_elements = 2
    epoches = 1
    for _ in tqdm(range(epoches), desc="Epochs", unit="epoch"):
        moduli = generate_crt_primes(limbs, crt_bitwidth)
        residues = generate_residues(moduli, poly_dim)
        # print(residues)
        # print(moduli)
        residues_original = copy.deepcopy(residues)
        moduli_out = group_multiply(moduli, limbs // limbs_out)
        # flip_n_bits_across_m_elements_2d_inplace(residues, flip_elements, flip_bits,
        #                                           # max_bit=crt_bitwidth
        #                                       )
        recon = bConv_ecc(residues, moduli, moduli_out)

