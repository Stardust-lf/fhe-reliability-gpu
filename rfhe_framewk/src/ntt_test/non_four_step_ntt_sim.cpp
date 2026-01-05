// mc_std_lazy_qbits.cpp
// ReliaFHE Simulation - Standard NTT with Lazy Reduction & Configurable Q Bits
// Features: 
//   - Configurable Q Bit-width (--q-bits)
//   - Standard Iterative NTT (No 4-Step)
//   - Lazy Reduction Check (Supportive IV)
//   - Global ABFT Check
//   - Multi-Fault Injection

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
int N_GLOBAL = 1024;
int TARGET_BIT_WIDTH_GLOBAL = 30; // Default 30-bit Q

u64 Q = 0;
u64 G = 0;
int BITS = 0;

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
    
    // Search downwards from limit
    for (u64 q = start; q > (1ULL << (bit_width - 1)); q -= step) {
        if (is_prime(q)) return q;
    }
    
    // If not found, try searching upwards (for very small bit widths)
    if (bit_width < 20) {
         start = (1ULL << (bit_width - 1)) / step * step + 1;
         for (u64 q = start; q < limit; q += step) {
             if (q > (1ULL << (bit_width - 1)) && is_prime(q)) return q;
         }
    }

    throw std::runtime_error("No suitable prime found for N=" + std::to_string(n_size) + 
                             " and bits=" + std::to_string(bit_width));
}

u64 find_primitive_root(u64 q) {
    if (q == 2) return 1;
    for (u64 g = 2; g < 100; ++g) {
        if (mod_pow(g, (q - 1) / 2, q) != 1) return g;
    }
    return 3;
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

// ---------- LAZY REDUCTION CHECKER ----------
class LazyChecker {
    int num_buckets;
    std::vector<u128> sum_ref; // A*W (True)
    std::vector<u128> sum_out; // V (Actual)
    
public:
    LazyChecker(int buckets) : num_buckets(buckets) {
        sum_ref.resize(buckets, 0);
        sum_out.resize(buckets, 0);
    }
    
    void accumulate(long long op_id, u64 v_actual, u128 v_true_absolute) {
        int b = op_id % num_buckets;
        sum_out[b] += v_actual;
        sum_ref[b] += v_true_absolute;
    }
    
    bool verify() {
        for (int i = 0; i < num_buckets; ++i) {
            if ((sum_out[i] % Q) != (sum_ref[i] % Q)) return false;
        }
        return true;
    }
};

using InjectionPlan = std::vector<std::pair<long long, int>>;

// ---------- Standard NTT with Lazy Check ----------
std::pair<long long, bool> ntt_standard_lazy(std::vector<u64>& A, 
                                             const InjectionPlan& plan, 
                                             ThreadRNG& rng, 
                                             bool enable_injection) {
    int n = A.size();
    long long op_idx = 0;
    int plan_ptr = 0;
    
    int buckets = (int)sqrt(n);
    if (buckets < 1) buckets = 1;
    LazyChecker lazy(buckets);

    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) std::swap(A[i], A[j]);
    }

    u64 root = mod_pow(G, (Q - 1) / n, Q);
    int length = 2;
    while (length <= n) {
        u64 wlen = mod_pow(root, n / length, Q);
        for (int i = 0; i < n; i += length) {
            u64 w = 1;
            int half = length / 2;
            for (int k = i; k < i + half; ++k) {
                u64 u = A[k];
                
                u128 product_abs = (u128)A[k + half] * w; 
                u64 v_true = product_abs % Q;
                u64 v = v_true;

                if (enable_injection) {
                    while (plan_ptr < plan.size() && plan[plan_ptr].first == op_idx) {
                        v = inject_one(v, plan[plan_ptr].second, rng);
                        plan_ptr++;
                    }
                }
                
                lazy.accumulate(op_idx, v, product_abs);
                op_idx++; 

                A[k] = (u + v) % Q;
                A[k + half] = (u + Q - v) % Q;
                w = (u128)w * wlen % Q;
            }
        }
        length <<= 1;
    }
    
    bool check_passed = lazy.verify();
    return {op_idx, check_passed};
}

// ---------- Global ABFT Check ----------
bool check_abft_global(const std::vector<u64>& input, const std::vector<u64>& output, ThreadRNG& rng) {
    int n = input.size();
    std::vector<u64> w(n);
    for(int i=0; i<n; ++i) w[i] = rng.rand_q();
    
    std::vector<u64> w_hat = w;
    InjectionPlan empty;
    ntt_standard_lazy(w_hat, empty, rng, false);
    
    u128 lhs = 0, rhs = 0;
    for(int i=0; i<n; ++i) {
        lhs = (lhs + (u128)w_hat[i] * input[i]) % Q;
        rhs = (rhs + (u128)w[i] * output[i]) % Q;
    }
    return lhs == rhs;
}

// ---------- Main ----------
int main(int argc, char** argv) {
    int trials = 10000;
    
    for(int i=1; i<argc; i++) {
        std::string s = argv[i];
        if (s == "--trials" && i+1 < argc) trials = std::stoi(argv[++i]);
        else if (s == "--N" && i+1 < argc) N_GLOBAL = std::stoi(argv[++i]);
        else if (s == "--q-bits" && i+1 < argc) TARGET_BIT_WIDTH_GLOBAL = std::stoi(argv[++i]);
    }

    try {
        Q = find_ntt_prime(N_GLOBAL, TARGET_BIT_WIDTH_GLOBAL);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    G = find_primitive_root(Q);
    u64 tmp = Q; BITS = 0; while(tmp>0) { tmp>>=1; BITS++; }
    
    std::cout << "Config: Standard NTT (N=" << N_GLOBAL << "), Q=" << Q 
              << " (" << BITS << "-bit)\n";
    std::cout << "Check Logic: Lazy Reduction (Mod Q) + Global ABFT\n";
    std::cout << "Running Multi-Fault Analysis...\n";
    std::cout << std::string(90, '=') << std::endl;

    struct Scenario { std::string name; std::vector<int> faults; };
    std::vector<Scenario> scenarios = {
        {"SBF", {SBF}}, {"DBF", {DBF}}, {"MLF", {MLF}},
        {"SBF+SBF", {SBF, SBF}}, {"SBF+DBF", {SBF, DBF}},
        {"MLF+SBF", {MLF, SBF}}, {"MLF+DBF", {MLF, DBF}}, {"MLF+MLF", {MLF, MLF}}
    };

    ThreadRNG drng(42);
    InjectionPlan empty;
    std::vector<u64> dummy(N_GLOBAL);
    auto dry_res = ntt_standard_lazy(dummy, empty, drng, false);
    long long total_ops = dry_res.first;

    std::cout << std::setw(15) << "Scenario" 
              << std::setw(18) << "Target"
              << std::setw(12) << "Det Rate" 
              << std::setw(12) << "Miss Rate"
              << " | " 
              << std::setw(12) << "LazyCatch" 
              << std::setw(12) << "InterCatch" << std::endl;
    std::cout << std::string(90, '-') << std::endl;

    for (auto& sc : scenarios) {
        std::atomic<long long> det_count{0};
        std::atomic<long long> c_lazy{0}, c_inter{0};

        #pragma omp parallel
        {
            ThreadRNG rng(2025 + omp_get_thread_num());
            
            #pragma omp for
            for (int t = 0; t < trials; ++t) {
                std::vector<u64> input(N_GLOBAL);
                for(int i=0; i<N_GLOBAL; ++i) input[i] = rng.rand_q();
                
                InjectionPlan plan;
                std::set<long long> used_ops;
                for (int type : sc.faults) {
                    long long target;
                    do { target = rng.rand_full() % total_ops; } while (used_ops.count(target));
                    used_ops.insert(target);
                    plan.push_back({target, type});
                }
                std::sort(plan.begin(), plan.end());

                std::vector<u64> output = input;
                auto res = ntt_standard_lazy(output, plan, rng, true);
                bool lazy_ok = res.second;

                bool inter_ok = check_abft_global(input, output, rng);

                bool detected = (!lazy_ok) || (!inter_ok);
                
                if (detected) {
                    det_count++;
                    if (!lazy_ok) c_lazy++;
                    if (!inter_ok) c_inter++;
                }
            }
        }

        double det_rate = (double)det_count / trials;
        double miss_rate = 1.0 - det_rate;
        
        std::cout << std::setw(15) << sc.name 
                  << std::setw(18) << "Standard NTT"
                  << std::fixed << std::setprecision(6) 
                  << std::setw(12) << det_rate 
                  << std::setw(12) << miss_rate
                  << " | " 
                  << std::setw(12) << c_lazy 
                  << std::setw(12) << c_inter << std::endl;
    }
    std::cout << std::string(90, '=') << std::endl;
    
    return 0;
}
