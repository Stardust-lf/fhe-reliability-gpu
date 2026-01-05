import argparse, math, random
from typing import Dict, Tuple, List
import numpy as np

# ---------- 1. 参数配置 ----------

# 目标: 30-bit 素数, N=4096 (需要 q-1 能被 8192 整除以支持 4-step 的子变换)
TARGET_BIT_WIDTH = 30
N_DEFAULT = 4096

def is_prime(n):
    if n % 2 == 0: return False
    # 简单的素性测试 (Miller-Rabin for deterministic range or strict check)
    # 对于 30-bit 整数，简单试除或 python内置 pow 即可验证
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in [2, 7, 61]: # bases for 32-bit int
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def find_ntt_prime(n_size, bit_width):
    # 寻找小于 2^bit_width 的最大 NTT 友好素数
    # q = k * 2N + 1 (为了安全通常找 2N，这里 N=4096, 找 k*8192+1)
    limit = 1 << bit_width
    step = 2 * n_size 
    start = (limit // step) * step + 1
    if start > limit: start -= step
    
    for q_cand in range(start, 1 << (bit_width-1), -step):
        if is_prime(q_cand):
            return q_cand
    raise ValueError("No suitable prime found")

def find_primitive_root(q):
    # 寻找原根
    if q == 2: return 1
    p1 = 2
    p2 = (q-1) // 2
    # 简单的假设 q = k*2^n + 1 的形式，尝试小整数
    for g in range(2, 100):
        # 原根的阶必须是 q-1
        # 对于 NTT 素数，主要是检查 g^((q-1)/2) != 1
        if pow(g, (q-1)//2, q) != 1:
            # 严谨的话要检查所有质因数，但对于 NTT 只要满足阶包含 N 即可
            # 这里简单返回一个生成元
            return g
    return 3

# 自动计算参数
q = find_ntt_prime(N_DEFAULT, TARGET_BIT_WIDTH)
g = find_primitive_root(q)

# 设置 24-bit 折叠模数 (2^24 + 1)
FOLD_MOD = (1 << 24) + 1 

BITS = q.bit_length()

# ---------- 2. 辅助函数 ----------
def mod_pow(a: int, e: int, m: int = q) -> int:
    return pow(a, e, m)

def root_of_unity(N: int) -> int:
    assert (q - 1) % N == 0, f"N={N} must divide q-1 ({q-1})"
    return mod_pow(g, (q - 1) // N, q)

def flip_bit_val(x: int, b: int) -> int:
    return (x ^ (1 << b)) % q

def flip_two_bits_val(x: int, b1: int, b2: int) -> int:
    if b1 == b2: return flip_bit_val(x, b1)
    return (x ^ (1 << b1) ^ (1 << b2)) % q

def inject_one(val: int, kind: str) -> int:
    """注入单一故障"""
    if kind == "SBF":
        b = random.randrange(BITS)
        return flip_bit_val(val, b)
    if kind == "DBF":
        b1 = random.randrange(BITS); b2 = random.randrange(BITS)
        return flip_two_bits_val(val, b1, b2)
    if kind == "MOF1":
        return random.randrange(q)
    raise ValueError("unknown fault kind")

# ---------- 3. 核心计算核心 (NTT & Twiddle) ----------

def ntt_inplace(vec: List[int], root: int,
                inj_plan: Dict[int, Tuple[str]] = None,
                op_base_idx: int = 0) -> int:
    """标准 NTT"""
    A = vec
    n = len(A)
    # 位反转
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
                # 蝶形乘法
                v_true = (A[j + half] * w) % q
                v = v_true
                
                if inj_plan and op_idx in inj_plan:
                    kind = inj_plan[op_idx][0]
                    v = inject_one(v, kind)
                op_idx += 1
                
                A[j]        = (u + v) % q
                A[j + half] = (u - v) % q
                w = (w * wlen) % q
        length <<= 1
    return op_idx

def twiddle_mul_inplace(M: np.ndarray, T: np.ndarray,
                        inj_plan: Dict[int, Tuple[str]] = None,
                        op_base_idx: int = 0) -> Tuple[int, bool]:
    """
    Twiddle 乘法 + Intra-element (24-bit Folding) 模拟
    """
    n2, n1 = M.shape
    op_idx = op_base_idx
    intra_detected_any = False 
    
    for r in range(n2):
        Tr = T[r]
        Mr = M[r]
        for c in range(n1):
            # 1. 真实计算
            v_true = (int(Mr[c]) * int(Tr[c])) % q
            v = v_true
            
            # 2. 注入故障
            if inj_plan and op_idx in inj_plan:
                kind = inj_plan[op_idx][0]
                v = inject_one(v, kind)
            op_idx += 1
            
            # 3. Intra-element 校验模拟 (Folding Check)
            if v != v_true:
                # 检查是否发生碰撞：模 (2^24 + 1)
                # 如果误差是 2^24+1 的倍数，这里会漏检 (return False)
                if (v % FOLD_MOD) != (v_true % FOLD_MOD):
                    intra_detected_any = True
                else:
                    # 只有 DBF(0, 24) 或类似组合会走到这里
                    # print(f"[DEBUG] Intra-check collision! v={v}, v_true={v_true}")
                    pass

            Mr[c] = v
            
    return op_idx, intra_detected_any

# ---------- 4. 保护校验逻辑 (ABFT) ----------

def batch_check_cols(A: np.ndarray, B: np.ndarray, n2_root: int) -> bool:
    """Batch-1 校验"""
    s_in  = np.sum(A, axis=1) % q
    s_out = np.sum(B, axis=1) % q
    w = [random.randrange(q) for _ in range(len(s_in))]
    w_hat = w.copy()
    ntt_inplace(w_hat, n2_root, None, 0)
    
    lhs = int(np.dot(w_hat, s_in) % q)
    rhs = int(np.dot(w,     s_out) % q)
    return lhs == rhs

def twiddle_check_inter(B_before: np.ndarray, B_after: np.ndarray, T: np.ndarray) -> bool:
    """Twiddle 阶段 Inter-element 校验"""
    n2, n1 = B_before.shape
    phi = np.array([random.randrange(q) for _ in range(n2)], dtype=np.int64)
    
    lhs = 0
    rhs = 0
    for c in range(n1):
        # 严格 ABFT 逻辑
        # lhs = <phi, Output>
        lhs = (lhs + int(np.dot(phi, B_after[:, c]) % q)) % q
        
        # rhs = <phi * T, Input>
        weighted_phi = (phi * T[:, c]) % q
        rhs = (rhs + int(np.dot(weighted_phi, B_before[:, c]) % q)) % q
        
    return lhs == rhs

def batch_check_rows(B: np.ndarray, C: np.ndarray, n1_root: int) -> bool:
    """Batch-2 校验"""
    r_in  = np.sum(B, axis=0) % q
    r_out = np.sum(C, axis=0) % q
    w = [random.randrange(q) for _ in range(len(r_in))]
    w_hat = w.copy()
    ntt_inplace(w_hat, n1_root, None, 0)
    
    lhs = int(np.dot(w_hat, r_in) % q)
    rhs = int(np.dot(w,     r_out) % q)
    return lhs == rhs

# ---------- 5. 四步法主流程 ----------

def four_step_ntt_protected(a: List[int], inj_plan: Dict[int, Tuple[str]] = None) -> Tuple[List[int], Dict]:
    N = len(a)
    n1 = int(round(math.isqrt(N))); n2 = n1
    assert n1 * n2 == N, "N must be a perfect square"

    wN   = root_of_unity(N)
    w_n1 = root_of_unity(n1)
    w_n2 = root_of_unity(n2)

    A = np.array([a[c*n2 + r] for c in range(n1) for r in range(n2)],
                 dtype=np.int64).reshape(n1, n2).T

    # Batch 1
    B = A.copy()
    op_idx = 0
    for c in range(n1):
        col = B[:, c].tolist()
        op_idx = ntt_inplace(col, w_n2, inj_plan, op_idx)
        B[:, c] = np.array(col, dtype=np.int64)
    ok_batch1 = batch_check_cols(A, B, w_n2)

    # Twiddle
    T = np.empty((n2, n1), dtype=np.int64)
    for r in range(n2):
        wr = mod_pow(wN, r, q)
        val = 1
        for c in range(n1):
            T[r, c] = val
            val = (val * wr) % q

    B_before = B.copy()
    op_idx, intra_detected = twiddle_mul_inplace(B, T, inj_plan, op_idx)
    inter_detected = not twiddle_check_inter(B_before, B, T)

    # Batch 2
    C = B.copy()
    for r in range(n2):
        row = C[r, :].tolist()
        op_idx = ntt_inplace(row, w_n1, inj_plan, op_idx)
        C[r, :] = np.array(row, dtype=np.int64)
    ok_batch2 = batch_check_rows(B, C, w_n1)

    out = C.T.flatten().astype(int).tolist()
    
    # 只要任意一个检测器触发就算成功
    is_detected = (not ok_batch1) or intra_detected or inter_detected or (not ok_batch2)
    
    info = dict(
        batch1_ok=int(ok_batch1), 
        intra_ok=int(not intra_detected), # 注意：这里 intra_detected=True 意味着发现了错误，所以 ok=False
        inter_ok=int(not inter_detected),
        batch2_ok=int(ok_batch2), 
        total_ops=op_idx,
        detected = int(is_detected)
    )
    return out, info

# ---------- 6. 实验 ----------

def make_single_injection_plan(total_ops: int, kind: str) -> Dict[int, Tuple[str]]:
    i = random.randrange(total_ops)
    return {i: (kind,)}

def run_trials(N: int, trials: int, kind: str, seed: int = 0) -> Tuple[float, float, Dict]:
    random.seed(seed)
    np.random.seed(seed)
    
    a0 = [0] * N
    _, info0 = four_step_ntt_protected(a0, inj_plan={})
    total_ops = info0["total_ops"]

    detected_count = 0
    # 统计是哪个机制抓住了错误
    stats = {"intra_catch": 0, "inter_catch": 0, "batch_catch": 0}

    for _ in range(trials):
        a = [random.randrange(q) for _ in range(N)]
        plan = make_single_injection_plan(total_ops, kind)
        _, info = four_step_ntt_protected(a, inj_plan=plan)
        
        if info["detected"]:
            detected_count += 1
            # 简单统计：优先归因为 Intra (最细粒度)，然后 Inter，然后 Batch
            if info["intra_ok"] == 0:
                stats["intra_catch"] += 1
            elif info["inter_ok"] == 0:
                stats["inter_catch"] += 1
            else:
                stats["batch_catch"] += 1
            
    det_rate = detected_count / trials
    miss_rate = 1.0 - det_rate
    return det_rate, miss_rate, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kinds", type=str, default="SBF,DBF,MOF1")
    args = ap.parse_args()

    print(f"Configuration: N={args.N}")
    print(f"Found Prime q={q} (bit_len={q.bit_length()})")
    print(f"Intra-Fold-Mod={FOLD_MOD} (2^24 + 1)")
    
    kinds = [s.strip() for s in args.kinds.split(",") if s.strip()]
    
    for k in kinds:
        det, miss, stats = run_trials(args.N, args.trials, k, args.seed)
        print(f"Fault Type [{k}]: Det={det:.6f} Miss={miss:.6f}")
        print(f"    Breakdown: Intra={stats['intra_catch']}, Inter={stats['inter_catch']}, Batch={stats['batch_catch']}")

if __name__ == "__main__":
    main()
