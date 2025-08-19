// fused_resilient.cpp
// Build: g++ -O3 -std=gnu++17 fused_resilient.cpp -o fused_resilient
// Run  : ./fused_resilient > result.csv
//
// What it does (no CLI; all params at top):
//   - Sweep fold width k = 4..24
//   - Trials per k: TRIALS
//   - Modulus width: QBITS = 30 (prime-like q ~ 2^30)
//   - n = 128 elements per vector
// Injections:
//   * Mul stage (Algorithm 2): MOF1 (randomize one product), MOF2 (two products)
//   * Reduction stage (Algorithm 1): SBF (flip 1 bit in s or c), DBF (flip 2 bits in s or c)
// Outputs rows: strategy,mode,fold,miss_rate
//
// Notes:
//   - Uses Boost.Multiprecision for safe big-int products and Barrett math.
//   - Comments in English per your requirement.

#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <omp.h>
using namespace std;
using boost::multiprecision::cpp_int;

// ===================== Parameters =====================
static constexpr size_t N       = 128;   // vector length
static constexpr int    QBITS   = 30;    // prime-like modulus bit-width
static constexpr int    K_MIN   = 4;     // fold width sweep start
static constexpr int    K_MAX   = 24;    // fold width sweep end (inclusive)
static constexpr int    TRIALS  = 10000; // trials per fold width
static constexpr uint64_t SEED  = 1;     // RNG seed

// ===================== RNG helpers =====================
static inline uint64_t rnd64(mt19937_64& eng){ return eng(); }
static inline int rand_bit_in_range(mt19937_64& eng, int qbits){
    uniform_int_distribution<int> d(0, qbits-1); return d(eng);
}
static inline size_t rand_index(mt19937_64& eng, size_t n){
    uniform_int_distribution<size_t> d(0, n-1); return d(eng);
}

// ===================== Fold mod (2^k+1) =====================
cpp_int fold_mod_2k_plus_1_big(const cpp_int& X, int k){
    cpp_int M = (cpp_int(1) << k) + 1;
    cpp_int mask = (cpp_int(1) << k) - 1;
    cpp_int x = X, acc = 0;
    int sign = +1;
    while(x > 0){
        cpp_int seg = x & mask;
        acc += (sign>0 ? seg : -seg);
        acc %= M;
        x >>= k;
        sign = -sign;
    }
    acc %= M; if(acc < 0) acc += M;
    return acc;
}

// ===================== Barrett reduction =====================
struct Barrett {
    cpp_int q;
    int K;       // ceil(log2(q))
    cpp_int mu;  // floor(2^(2K) / q)
};
static int bitlen(const cpp_int& x){
    if(x==0) return 0;
    cpp_int t=x; int n=0; while(t>0){ t>>=1; ++n; } return n;
}
static Barrett make_barrett_from_qbits(int qbits){
    // simple odd near 2^qbits; replace with NTT-friendly prime in real systems
    cpp_int q = (cpp_int(1) << qbits) - 59; // arbitrary odd
    if ((q & 1)==0) q += 1;
    int K = bitlen(q);
    cpp_int twoK = cpp_int(1) << (2*K);
    cpp_int mu = twoK / q;
    return {q, K, mu};
}

// Flip one or two bits in low QBITS of big-int
static inline void flip_bit_low(cpp_int& v, int b){ v ^= (cpp_int(1) << b); }
static inline void inject_SBF_on(cpp_int& v, mt19937_64& eng){
    int b = rand_bit_in_range(eng, QBITS); flip_bit_low(v, b);
}
static inline void inject_DBF_on(cpp_int& v, mt19937_64& eng){
    int b1 = rand_bit_in_range(eng, QBITS), b2;
    do { b2 = rand_bit_in_range(eng, QBITS); } while(b2==b1);
    flip_bit_low(v, b1); flip_bit_low(v, b2);
}

// ===================== Algorithm 2: mul + intra =====================
// Return true if detected by intra checksum at any changed index.
static bool mul_intra_detect_MOF(const vector<uint64_t>& a,
                                 const vector<uint64_t>& b,
                                 const vector<size_t>& changed_idx,
                                 const vector<cpp_int>& t_fault,
                                 int k)
{
    cpp_int M = (cpp_int(1) << k) + 1;
    for(size_t idx : changed_idx){
        cpp_int ah = fold_mod_2k_plus_1_big(cpp_int(a[idx]), k);
        cpp_int bh = fold_mod_2k_plus_1_big(cpp_int(b[idx]), k);
        cpp_int th = fold_mod_2k_plus_1_big(t_fault[idx], k);
        if (th != (ah*bh)%M) return true; // detected
    }
    return false; // all changed passed → miss
}

// ===================== Algorithm 1: reduction + checks =====================
// Run Barrett reduction on full vector, but inject SBF/DBF on s or c at one random idx.
// Returns: pair{detected_by_range, detected_by_inter}
static pair<bool,bool> reduce_with_fault_and_check(const vector<cpp_int>& t_vec,
                                                   const Barrett& B,
                                                   mt19937_64& eng)
{
    size_t n = t_vec.size();
    vector<cpp_int> c(n);
    bool detected_range = false;

    // choose where to inject: s or c, at a random index
    bool target_s = uniform_int_distribution<int>(0,1)(eng)==0;
    size_t idx = rand_index(eng, n);

    cpp_int sum_c = 0;
    cpp_int sum_t = 0;
    const int K = B.K; const cpp_int q=B.q; const cpp_int mu=B.mu;

    for(size_t i=0;i<n;i++){
        const cpp_int& ti = t_vec[i];
        sum_t = (sum_t + (ti % q)) % q;

        // s = floor( (t * mu) / 2^(2K) )
        cpp_int s = (ti * mu) >> (2*K);

        // INJECTION: flip bits on s BEFORE forming c
        cpp_int s_used = s;
        if (i==idx && target_s){
            if (uniform_int_distribution<int>(0,1)(eng)==0) inject_SBF_on(s_used, eng);
            else inject_DBF_on(s_used, eng);
        }

        // c = t - s*q
        cpp_int ci = ti - s_used * q;

        // INJECTION: flip bits on c BEFORE final range-fix
        if (i==idx && !target_s){
            if (uniform_int_distribution<int>(0,1)(eng)==0) inject_SBF_on(ci, eng);
            else inject_DBF_on(ci, eng);
        }

        if (ci >= q) ci -= q;           // one subtract
        if (!(ci >= 0 && ci < q)) {     // range check
            detected_range = true;
        }

        c[i] = ci;
        sum_c = (sum_c + ci) % q;
    }
    bool detected_inter = (sum_c % B.q) != (sum_t % B.q);
    return {detected_range, detected_inter};
}

// ===================== Trial runner per k =====================
struct Counters {
    uint64_t miss_mul_intra_MOF1 = 0;
    uint64_t miss_mul_intra_MOF2 = 0;
    uint64_t miss_red_range_SBF  = 0;
    uint64_t miss_red_inter_SBF  = 0;
    uint64_t miss_red_range_DBF  = 0;
    uint64_t miss_red_inter_DBF  = 0;
};


static void run_for_k(int k, Counters& C, mt19937_64& /*eng_base*/) {
    auto rand_big128 = [](mt19937_64& eng)->cpp_int {
        cpp_int r = 0;
        r <<= 64; r += eng();
        r <<= 64; r += eng();
        return r;
    };

    const cpp_int M = (cpp_int(1) << k) + 1;
    const auto B = make_barrett_from_qbits(QBITS);
    const int  Kb = B.K;
    const cpp_int q  = B.q;
    const cpp_int mu = B.mu;
    const cpp_int q_hat  = fold_mod_2k_plus_1_big(q,  k);
    const cpp_int mu_hat = fold_mod_2k_plus_1_big(mu, k);

    uint64_t miss_mof1 = 0, miss_mof2 = 0;
    uint64_t miss_rr_sbf = 0, miss_ri_sbf = 0;
    uint64_t miss_rr_dbf = 0, miss_ri_dbf = 0;

    #pragma omp parallel reduction(+:miss_mof1,miss_mof2,miss_rr_sbf,miss_ri_sbf,miss_rr_dbf,miss_ri_dbf)
    {
        uint64_t seed = SEED ^ (uint64_t)k ^ (0x9E3779B97F4A7C15ULL *
        #ifdef _OPENMP
            (uint64_t)omp_get_thread_num()
        #else
            0ULL
        #endif
        );
        mt19937_64 eng(seed);

        #pragma omp for schedule(static)
        for (int t = 0; t < TRIALS; ++t) {
            // ----- inputs & true products -----
            vector<uint64_t> a(N), b(N);
            for (size_t i = 0; i < N; ++i) { a[i] = eng(); b[i] = eng(); }
            vector<cpp_int> t_true(N);
            for (size_t i = 0; i < N; ++i) t_true[i] = cpp_int(a[i]) * cpp_int(b[i]);

            // ===== MOF1 =====
            {
                vector<cpp_int> t_fault = t_true;
                size_t idx = rand_index(eng, N);
                t_fault[idx] = rand_big128(eng);
                // intra check at changed indices only
                bool detected = false;
                {
                    cpp_int ah = fold_mod_2k_plus_1_big(cpp_int(a[idx]), k);
                    cpp_int bh = fold_mod_2k_plus_1_big(cpp_int(b[idx]), k);
                    cpp_int th = fold_mod_2k_plus_1_big(t_fault[idx], k);
                    detected = (th == (ah * bh) % M) ? false : true;
                }
                if (!detected) ++miss_mof1;
            }
            // ===== MOF2 =====
            {
                vector<cpp_int> t_fault = t_true;
                size_t i1 = rand_index(eng, N), i2;
                do { i2 = rand_index(eng, N); } while (i2 == i1);
                t_fault[i1] = rand_big128(eng);
                t_fault[i2] = rand_big128(eng);
                bool detected1 = false, detected2 = false;
                {
                    cpp_int ah = fold_mod_2k_plus_1_big(cpp_int(a[i1]), k);
                    cpp_int bh = fold_mod_2k_plus_1_big(cpp_int(b[i1]), k);
                    cpp_int th = fold_mod_2k_plus_1_big(t_fault[i1], k);
                    detected1 = (th != (ah * bh) % M);
                }
                {
                    cpp_int ah = fold_mod_2k_plus_1_big(cpp_int(a[i2]), k);
                    cpp_int bh = fold_mod_2k_plus_1_big(cpp_int(b[i2]), k);
                    cpp_int th = fold_mod_2k_plus_1_big(t_fault[i2], k);
                    detected2 = (th != (ah * bh) % M);
                }
                if (!(detected1 || detected2)) ++miss_mof2;
            }

            // ===== Reduction with SBF =====
            {
                size_t n = t_true.size();
                vector<cpp_int> c(n);
                bool detected_range = false;

                bool target_s = uniform_int_distribution<int>(0,1)(eng) == 0;
                size_t idx = rand_index(eng, n);

                cpp_int sum_c = 0, sum_t = 0;

                for (size_t i = 0; i < n; ++i) {
                    const cpp_int& ti = t_true[i];
                    sum_t = (sum_t + (ti % q)) % q;

                    // u = t * mu, check: fold(u) ?= fold(t)*fold(mu)
                    cpp_int u = ti * mu;
                    cpp_int u_hat = fold_mod_2k_plus_1_big(u, k);
                    cpp_int expect_u = (fold_mod_2k_plus_1_big(ti, k) * mu_hat) % M;
                    // if needed, you can统计：bool mul1_ok = (u_hat == expect_u);

                    // s
                    cpp_int s = u >> (2 * Kb);
                    cpp_int s_used = s;
                    if (i == idx && target_s) inject_SBF_on(s_used, eng);

                    // v = s_used * q, check: fold(v) ?= fold(s_used)*fold(q)
                    cpp_int v = s_used * q;
                    cpp_int v_hat = fold_mod_2k_plus_1_big(v, k);
                    cpp_int expect_v = (fold_mod_2k_plus_1_big(s_used, k) * q_hat) % M;
                    // bool mul2_ok = (v_hat == expect_v);

                    // c
                    cpp_int ci = ti - v;
                    if (i == idx && !target_s) inject_SBF_on(ci, eng);

                    // robust correction
                    if (ci >= q) { ci -= q; if (ci >= q) ci -= q; }
                    else if (ci < 0) { ci += q; if (ci < 0) ci += q; }

                    if (!(ci >= 0 && ci < q)) detected_range = true;

                    c[i] = ci;
                    sum_c = (sum_c + ci) % q;
                }
                bool detected_inter = (sum_c % q) != (sum_t % q);
                if (!detected_range) ++miss_rr_sbf;
                if (!detected_inter) ++miss_ri_sbf;
            }

            // ===== Reduction with DBF =====
            {
                size_t n = t_true.size();
                vector<cpp_int> c(n);
                bool detected_range = false;

                bool target_s = uniform_int_distribution<int>(0,1)(eng) == 0;
                size_t idx = rand_index(eng, n);

                cpp_int sum_c = 0, sum_t = 0;

                for (size_t i = 0; i < n; ++i) {
                    const cpp_int& ti = t_true[i];
                    sum_t = (sum_t + (ti % q)) % q;

                    cpp_int u = ti * mu;
                    cpp_int u_hat = fold_mod_2k_plus_1_big(u, k);
                    cpp_int expect_u = (fold_mod_2k_plus_1_big(ti, k) * mu_hat) % M;

                    cpp_int s = u >> (2 * Kb);
                    cpp_int s_used = s;
                    if (i == idx && target_s) inject_DBF_on(s_used, eng);

                    cpp_int v = s_used * q;
                    cpp_int v_hat = fold_mod_2k_plus_1_big(v, k);
                    cpp_int expect_v = (fold_mod_2k_plus_1_big(s_used, k) * q_hat) % M;

                    cpp_int ci = ti - v;
                    if (i == idx && !target_s) inject_DBF_on(ci, eng);

                    if (ci >= q) { ci -= q; if (ci >= q) ci -= q; }
                    else if (ci < 0) { ci += q; if (ci < 0) ci += q; }

                    if (!(ci >= 0 && ci < q)) detected_range = true;

                    c[i] = ci;
                    sum_c = (sum_c + ci) % q;
                }
                bool detected_inter = (sum_c % q) != (sum_t % q);
                if (!detected_range) ++miss_rr_dbf;
                if (!detected_inter) ++miss_ri_dbf;
            }
        } // TRIALS
    } // parallel

    C.miss_mul_intra_MOF1 += miss_mof1;
    C.miss_mul_intra_MOF2 += miss_mof2;
    C.miss_red_range_SBF  += miss_rr_sbf;
    C.miss_red_inter_SBF  += miss_ri_sbf;
    C.miss_red_range_DBF  += miss_rr_dbf;
    C.miss_red_inter_DBF  += miss_ri_dbf;
}


// ===================== Main =====================
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    mt19937_64 eng(SEED);
    cout << "strategy,mode,fold,miss_rate\n";

    for(int k=K_MIN; k<=K_MAX; k+=2){
        Counters C{};
        run_for_k(k, C, eng);

        auto emit = [&](const string& strat, const string& mode, uint64_t miss){
            double mr = double(miss) / double(TRIALS);
            cout << strat << "," << mode << "," << k << "," << setprecision(17) << mr << "\n";
        };

        // Mul stage results (only meaningful for MOF modes)
        emit("MulIntra","MOF1", C.miss_mul_intra_MOF1);
        emit("MulIntra","MOF2", C.miss_mul_intra_MOF2);

        // Reduction stage results (only meaningful for SBF/DBF)
        emit("RedRange","SBF",  C.miss_red_range_SBF);
        emit("RedInter","SBF",  C.miss_red_inter_SBF);
        emit("RedRange","DBF",  C.miss_red_range_DBF);
        emit("RedInter","DBF",  C.miss_red_inter_DBF);
    }
    return 0;
}

