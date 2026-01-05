// mc_ntt_multi_stages_qbits.cpp
// ReliaFHE Simulation - Multi-Fault Protocols with Stage-wise Breakdown & Configurable Q
// Features: SBF/DBF/MLF Combinations, Stage-specific Injection, Detailed Detection Stats, Q-Bits Config

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdint>
#include <algorithm>
#include <map>
#include <iomanip>
#include <omp.h>
#include <cassert>
#include <atomic>
#include <cstring>
#include <string>
#include <set>

using u64 = uint64_t;
using u128 = __uint128_t;

// ---------- Configuration (Globals) ----------
int N_GLOBAL = 4096;
int FOLD_BITS_GLOBAL = 4;
int TARGET_BIT_WIDTH_GLOBAL = 30; // Default 30-bit

// Global computed params
u64 Q = 0;
u64 G = 0;
int BITS = 0;
u64 FOLD_MASK = 0;

// ---------- Helpers ----------
u64 mod_pow(u64 base, u64 exp, u64 mod) {
    u64 res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (u128)res * base % mod;
        base = (u128)base * base % mod;
        exp /= 2;
    }
    return res;
}

bool is_prime(u64 n) {
    if (n < 2) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;
    u64 d = n - 1;
    int r = 0;
    while (d % 2 == 0) { d /= 2; r++; }
    for (u64 a : {2, 7, 61}) {
        if (a >= n) break;
        u64 x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool passed = false;
        for (int i = 0; i < r - 1; ++i) {
            x = (u128)x * x % n;
            if (x == n - 1) { passed = true; break; }
        }
        if (!passed) return false;
    }
    return true;
}

u64 find_ntt_prime(int n_size, int bit_width) {
    u64 limit = 1ULL << bit_width;
    u64 step = 2ULL * n_size; 
    u64 start = (limit / step) * step + 1;
    if (start > limit) start -= step;
    
    // Search downwards
    for (u64 q = start; q > (1ULL << (bit_width - 1)); q -= step) {
        if (is_prime(q)) return q;
    }
    
    // Fallback search up (for small bit widths)
    if (bit_width < 20) {
         start = (1ULL << (bit_width - 1)) / step * step + 1;
         for (u64 q = start; q < limit; q += step) {
             if (q > (1ULL << (bit_width - 1)) && is_prime(q)) return q;
         }
    }
    throw std::runtime_error("No suitable prime found");
}

u64 find_primitive_root(u64 q) {
    if (q == 2) return 1;
    for (u64 g = 2; g < 100; ++g) {
        if (mod_pow(g, (q - 1) / 2, q) != 1) return g;
    }
    return 3;
}

u64 root_of_unity(int N) {
    return mod_pow(G, (Q - 1) / N, Q);
}

// ---------- RNG & Injection ----------
struct ThreadRNG {
    std::mt19937_64 rng;
    ThreadRNG(u64 seed) : rng(seed) {}
    u64 rand_q() { return std::uniform_int_distribution<u64>(0, Q - 1)(rng); }
    u64 rand_full() { return rng(); }
    int rand_bits() { return std::uniform_int_distribution<int>(0, BITS - 1)(rng); }
};

enum FaultType {
    NONE = 0,
    SBF = 1,
    DBF = 2,
    MLF = 3
};

u64 flip_bit_val(u64 x, int b) { return (x ^ (1ULL << b)) % Q; }

u64 inject_one(u64 val, int kind, ThreadRNG& rng) {
    if (kind == SBF) { 
        return flip_bit_val(val, rng.rand_bits()); 
    }
    if (kind == DBF) { 
        int b1 = rng.rand_bits();
        int b2 = rng.rand_bits();
        while (b1 == b2) { b2 = rng.rand_bits(); } 
        return (val ^ (1ULL << b1) ^ (1ULL << b2)) % Q;
    }
    if (kind == MLF) { 
        return rng.rand_full() % Q; 
    }
    return val;
}

// ---------- Hardware Folding Logic ----------
inline u64 calc_fold_checksum(u64 val, int w) {
    u64 sum = 0;
    while (val > 0) {
        sum += (val & FOLD_MASK);
        val >>= w;
    }
    return sum & FOLD_MASK;
}

// ---------- Structures ----------
using InjectionPlan = std::vector<std::pair<long long, int>>;

struct OpRanges {
    long long s1_start; long long s1_end;
    long long s2_start; long long s2_end;
    long long s3_start; long long s3_end;
    long long total;
};

struct TrialStats {
    bool ok_b1;
    bool ok_intra;
    bool ok_inter;
    bool ok_b2;
    bool detected;
    OpRanges ranges;
};

struct TwiddleResult {
    long long ops;
    bool intra_detected;
};

// ---------- NTT Logic ----------
int ntt_inplace(std::vector<u64>& A, u64 root, 
                long long& op_idx, const InjectionPlan& plan, int& plan_ptr, ThreadRNG& rng) {
    int n = A.size();
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(A[i], A[j]);
    }

    int length = 2;
    while (length <= n) {
        u64 wlen = mod_pow(root, n / length, Q);
        for (int i = 0; i < n; i += length) {
            u64 w = 1;
            int half = length / 2;
            for (int k = i; k < i + half; ++k) {
                u64 u = A[k];
                u64 v_true = (u128)A[k + half] * w % Q;
                u64 v = v_true;

                // Check injection plan
                while (plan_ptr < plan.size() && plan[plan_ptr].first == op_idx) {
                    v = inject_one(v, plan[plan_ptr].second, rng);
                    plan_ptr++;
                }
                op_idx++;

                A[k] = (u + v) % Q;
                A[k + half] = (u + Q - v) % Q;
                w = (u128)w * wlen % Q;
            }
        }
        length <<= 1;
    }
    return 0; 
}

TwiddleResult twiddle_mul_inplace(std::vector<u64>& M, const std::vector<u64>& T, 
                                  long long start_op, const InjectionPlan& plan, int& plan_ptr, ThreadRNG& rng) {
    long long op_idx = start_op;
    bool intra_detected = false;
    
    for (size_t i = 0; i < M.size(); ++i) {
        u64 v_true = (u128)M[i] * T[i] % Q;
        u64 v = v_true;
        
        while (plan_ptr < plan.size() && plan[plan_ptr].first == op_idx) {
            v = inject_one(v, plan[plan_ptr].second, rng);
            plan_ptr++;
        }
        op_idx++;
        
        // Intra Check (Hardware Fold)
        if (v != v_true) {
            u64 chk_true = calc_fold_checksum(v_true, FOLD_BITS_GLOBAL);
            u64 chk_fault = calc_fold_checksum(v, FOLD_BITS_GLOBAL);
            if (chk_true != chk_fault) intra_detected = true;
        }
        M[i] = v;
    }
    return {op_idx, intra_detected};
}

// ---------- Protection Checks ----------
bool batch_check(const std::vector<u64>& In, const std::vector<u64>& Out, 
                 int n_sub, int len_sub, u64 root, ThreadRNG& rng, bool check_cols) {
    std::vector<u64> s_in(len_sub, 0);
    std::vector<u64> s_out(len_sub, 0);

    if (check_cols) { 
        int n1 = n_sub; int n2 = len_sub;
        for(int r=0; r<n2; ++r) {
            u128 sum_i=0, sum_o=0;
            for(int c=0; c<n1; ++c) {
                sum_i += In[c*n2+r]; sum_o += Out[c*n2+r];
            }
            s_in[r] = sum_i % Q; s_out[r] = sum_o % Q;
        }
    } else { 
        int n2 = n_sub; int n1 = len_sub;
        for(int c=0; c<n1; ++c) {
            u128 sum_i=0, sum_o=0;
            for(int r=0; r<n2; ++r) {
                sum_i += In[r*n1+c]; sum_o += Out[r*n1+c];
            }
            s_in[c] = sum_i % Q; s_out[c] = sum_o % Q;
        }
    }

    std::vector<u64> w(len_sub);
    for(int i=0; i<len_sub; ++i) w[i] = rng.rand_q();
    
    std::vector<u64> w_hat = w;
    // Dummy calls for w_hat transform
    InjectionPlan empty; int ptr=0; long long dummy_op=-1;
    ntt_inplace(w_hat, root, dummy_op, empty, ptr, rng);
    
    u128 lhs = 0, rhs = 0;
    for(int i=0; i<len_sub; ++i) {
        lhs = (lhs + (u128)w_hat[i] * s_in[i]) % Q;
        rhs = (rhs + (u128)w[i] * s_out[i]) % Q;
    }
    return lhs == rhs;
}

bool check_inter(const std::vector<u64>& B_pre, const std::vector<u64>& B_post, 
                 const std::vector<u64>& T, int n1, int n2, ThreadRNG& rng) {
    std::vector<u64> phi(n2);
    for(int i=0; i<n2; ++i) phi[i] = rng.rand_q();
    u128 lhs=0, rhs=0;
    for(int c=0; c<n1; ++c) {
        u128 lc=0, rc=0;
        for(int r=0; r<n2; ++r) {
            int idx = c*n2+r;
            lc = (lc + (u128)phi[r]*B_post[idx]) % Q;
            u64 w_phi = (u128)phi[r]*T[idx] % Q;
            rc = (rc + (u128)w_phi*B_pre[idx]) % Q;
        }
        lhs = (lhs+lc)%Q; rhs = (rhs+rc)%Q;
    }
    return lhs==rhs;
}

// ---------- 4-Step Flow ----------
TrialStats four_step_ntt(int N, const InjectionPlan& plan, ThreadRNG& rng) {
    int n1 = (int)round(sqrt(N));
    int n2 = N / n1;
    OpRanges r;
    int plan_ptr = 0;
    
    std::vector<u64> A(N);
    for(int i=0; i<N; ++i) A[i] = rng.rand_q();

    // Batch 1
    std::vector<u64> B = A;
    long long op_idx = 0;
    u64 w_n2 = root_of_unity(n2);
    
    r.s1_start = op_idx;
    for (int c = 0; c < n1; ++c) {
        std::vector<u64> col(B.begin() + c*n2, B.begin() + (c+1)*n2);
        ntt_inplace(col, w_n2, op_idx, plan, plan_ptr, rng);
        std::copy(col.begin(), col.end(), B.begin() + c*n2);
    }
    r.s1_end = op_idx;
    bool ok_b1 = batch_check(A, B, n1, n2, w_n2, rng, true);

    // Twiddle
    u64 wN = root_of_unity(N);
    std::vector<u64> T(N);
    for (int r = 0; r < n2; ++r) {
        u64 wr = mod_pow(wN, r, Q);
        u64 val = 1;
        for (int c = 0; c < n1; ++c) {
            T[c * n2 + r] = val;
            val = (u128)val * wr % Q;
        }
    }
    
    r.s2_start = op_idx;
    std::vector<u64> B_pre = B;
    TwiddleResult tw_res = twiddle_mul_inplace(B, T, op_idx, plan, plan_ptr, rng);
    op_idx = tw_res.ops;
    r.s2_end = op_idx;
    
    bool ok_inter = check_inter(B_pre, B, T, n1, n2, rng);

    // Batch 2
    std::vector<u64> C_transposed(N);
    for (int c = 0; c < n1; ++c) {
        for (int r = 0; r < n2; ++r) {
            C_transposed[r * n1 + c] = B[c * n2 + r];
        }
    }
    std::vector<u64> C_in = C_transposed;
    u64 w_n1 = root_of_unity(n1);
    
    r.s3_start = op_idx;
    for (int r = 0; r < n2; ++r) {
        std::vector<u64> row(C_transposed.begin() + r*n1, C_transposed.begin() + (r+1)*n1);
        ntt_inplace(row, w_n1, op_idx, plan, plan_ptr, rng);
        std::copy(row.begin(), row.end(), C_transposed.begin() + r*n1);
    }
    r.s3_end = op_idx;
    r.total = op_idx;
    
    bool ok_b2 = batch_check(C_in, C_transposed, n2, n1, w_n1, rng, false);

    bool detected = (!ok_b1) || tw_res.intra_detected || (!ok_inter) || (!ok_b2);
    return {ok_b1, !tw_res.intra_detected, ok_inter, ok_b2, detected, r};
}

// ---------- Main ----------
int main(int argc, char** argv) {
    int trials = 10000;
    
    for(int i=1; i<argc; i++) {
        std::string s = argv[i];
        if (s == "--trials" && i+1 < argc) trials = std::stoi(argv[++i]);
        else if (s == "--N" && i+1 < argc) N_GLOBAL = std::stoi(argv[++i]);
        else if (s == "--fold-bits" && i+1 < argc) FOLD_BITS_GLOBAL = std::stoi(argv[++i]);
        else if (s == "--q-bits" && i+1 < argc) TARGET_BIT_WIDTH_GLOBAL = std::stoi(argv[++i]);
    }

    FOLD_MASK = (1ULL << FOLD_BITS_GLOBAL) - 1;
    
    try {
        Q = find_ntt_prime(N_GLOBAL, TARGET_BIT_WIDTH_GLOBAL);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    G = find_primitive_root(Q);
    u64 tmp = Q; BITS = 0; while(tmp>0) { tmp>>=1; BITS++; }
    
    std::cout << "Config: N=" << N_GLOBAL << ", Fold=" << FOLD_BITS_GLOBAL 
              << " bits, Q=" << Q << " (" << BITS << "-bit)\n";
    std::cout << "Running Multi-Fault Protocol (Stage-wise)...\n";
    std::cout << std::string(95, '=') << std::endl;

    // Scenarios from Figure 10
    struct Scenario { 
        std::string name; 
        std::vector<int> faults; 
    };
    // std::vector<Scenario> scenarios = {
    //     {"SBF+SBF", {SBF, SBF}},
    //     {"SBF+DBF", {SBF, DBF}},
    // };
    std::vector<Scenario> scenarios = {
        {"SBF", {SBF}}, {"DBF", {DBF}}, {"MLF", {MLF}},
        {"SBF+SBF", {SBF, SBF}}, {"SBF+DBF", {SBF, DBF}},
        {"MLF+SBF", {MLF, SBF}}, {"MLF+DBF", {MLF, DBF}}, {"MLF+MLF", {MLF, MLF}}
    };

    // Dry Run for Ranges
    ThreadRNG drng(42);
    InjectionPlan empty;
    TrialStats dry = four_step_ntt(N_GLOBAL, empty, drng);
    OpRanges r = dry.ranges;
    
    struct StageDef { std::string name; long long start; long long end; };
    std::vector<StageDef> stages = {
        {"Stage1(Batch1)", r.s1_start, r.s1_end},
        {"Stage2(Twiddle)", r.s2_start, r.s2_end},
        {"Stage3(Batch2)", r.s3_start, r.s3_end}
    };

    for (auto& sc : scenarios) {
        std::cout << "Scenario: [" << sc.name << "]" << std::endl;
        std::cout << std::string(95, '-') << std::endl;
        std::cout << std::setw(18) << "Target Stage" 
                  << std::setw(12) << "Det Rate" 
                  << std::setw(12) << "Miss Rate"
                  << " | " 
                  << std::setw(10) << "Batch1" 
                  << std::setw(10) << "Intra" 
                  << std::setw(10) << "Inter" 
                  << std::setw(10) << "Batch2" << std::endl;

        for (auto& stg : stages) {
            std::atomic<long long> det_count{0};
            std::atomic<long long> c_b1{0}, c_intra{0}, c_inter{0}, c_b2{0};
            long long range_len = stg.end - stg.start;

            #pragma omp parallel
            {
                ThreadRNG rng(2025 + omp_get_thread_num());
                #pragma omp for
                for (int t = 0; t < trials; ++t) {
                    InjectionPlan plan;
                    std::set<long long> used_ops;
                    
                    // Generate Multi-Faults within the CURRENT Stage
                    for (int type : sc.faults) {
                        long long target;
                        do {
                            target = stg.start + (rng.rand_full() % range_len);
                        } while (used_ops.count(target));
                        used_ops.insert(target);
                        plan.push_back({target, type});
                    }
                    std::sort(plan.begin(), plan.end());
                    
                    TrialStats stats = four_step_ntt(N_GLOBAL, plan, rng);
                    
                    if (stats.detected) {
                        det_count++;
                        if (!stats.ok_b1) c_b1++;
                        if (!stats.ok_intra) c_intra++;
                        if (!stats.ok_inter) c_inter++;
                        if (!stats.ok_b2) c_b2++;
                    }
                }
            }

            double det_rate = (double)det_count / trials;
            double miss_rate = 1.0 - det_rate;
            
            std::cout << std::setw(18) << stg.name 
                      << std::fixed << std::setprecision(6) 
                      << std::setw(12) << det_rate 
                      << std::setw(12) << miss_rate
                      << " | " 
                      << std::setw(10) << c_b1 
                      << std::setw(10) << c_intra 
                      << std::setw(10) << c_inter 
                      << std::setw(10) << c_b2 << std::endl;
        }
        std::cout << std::string(95, '=') << std::endl;
    }
    
    return 0;
}
