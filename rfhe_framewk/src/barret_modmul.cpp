// g++ -O3 -std=c++17 -fopenmp barrett_mc.cpp -o barrett_mc
// Requires: Boost headers (header-only) for multiprecision

#include <bits/stdc++.h>
#include <omp.h>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using boost::multiprecision::cpp_int;

// -------------------- Hyperparameters --------------------
static constexpr uint32_t BITWIDTH_PRIME   = 37;
static constexpr size_t   VECTOR_LEN       = 8192;
static constexpr int      TRIALS           = 100000;
static const vector<int>  FOLD_WIDTHS      = []{
    vector<int> v; for(int w=2; w<=32; w+=2) v.push_back(w); return v;
}();
struct Scheme { bool use_t, use_sn, use_final; };
static const vector<Scheme> SCHEMES = {
    {true,  false, false},
    {false, true,  false},
    {false, false, true},
};
// Fault injection in two dimensions
static constexpr int INJECT_ELEM_COUNT = 1;   // how many elements corrupted per trial
static constexpr int BITFLIPS_PER_ELEM = 2;   // distinct bit flips per corrupted element
// ---------------------------------------------------------

// -------------------- Utilities --------------------

// 64-bit Miller-Rabin primality test (deterministic for 64-bit)
static uint64_t mulmod_u64(uint64_t a, uint64_t b, uint64_t mod){
    __uint128_t r = ( (__uint128_t)a * b ) % mod;
    return (uint64_t)r;
}
static uint64_t powmod_u64(uint64_t a, uint64_t e, uint64_t mod){
    uint64_t r = 1;
    while(e){
        if(e & 1) r = mulmod_u64(r, a, mod);
        a = mulmod_u64(a, a, mod);
        e >>= 1;
    }
    return r;
}
static bool is_prime_u64(uint64_t n){
    if(n < 2) return false;
    static const uint64_t small[] = {2,3,5,7,11,13,17,19,23,29,31};
    for(uint64_t p: small){
        if(n%p==0) return n==p;
    }
    uint64_t d = n-1, s=0;
    while((d&1)==0){ d>>=1; ++s; }
    // Deterministic bases for 64-bit
    static const uint64_t bases[] = {2,3,5,7,11,13,17};
    for(uint64_t a: bases){
        if(a % n == 0) continue;
        uint64_t x = powmod_u64(a, d, n);
        if(x==1 || x==n-1) continue;
        bool comp = true;
        for(uint64_t r=1; r<s; ++r){
            x = mulmod_u64(x, x, n);
            if(x == n-1){ comp=false; break; }
        }
        if(comp) return false;
    }
    return true;
}

// Generate random prime with given bitwidth (<=64 bits here)
static uint64_t gen_prime_bitwidth(std::mt19937_64 &rng, uint32_t bitwidth){
    while(true){
        uint64_t p = 0;
        if(bitwidth >= 64){
            p = (rng() | (1ULL<<63)) | 1ULL; // force top bit and odd
        }else{
            uint64_t top = 1ULL << (bitwidth-1);
            uint64_t mask = (bitwidth==64)? ~0ULL : ((1ULL<<bitwidth)-1);
            p = (rng() & mask) | top | 1ULL;
        }
        if(is_prime_u64(p)) return p;
    }
}

// bit_length for cpp_int
static inline size_t bitlen(const cpp_int &x){
    if(x==0) return 0;
    return boost::multiprecision::msb(x) + 1;
}

// fold_mod: fold x into width-bit chunks modulo (2^width - 1)
static cpp_int fold_mod(const cpp_int &x_in, int width){
    if(width <= 0) throw std::runtime_error("width must be positive");
    cpp_int x = x_in;
    cpp_int M = (cpp_int(1) << width) - 1;
    cpp_int s = 0;
    while(x > 0){
        s += (x & M);
        x >>= width;
    }
    s %= M;
    return s;
}

// Flip bits of value at given positions (in-place return)
static inline cpp_int flip_bits(const cpp_int &v, const vector<size_t> &positions){
    cpp_int out = v;
    for(size_t pos: positions){
        out ^= (cpp_int(1) << pos);
    }
    return out;
}

// Inject faults on 'elem_count' distinct indices, flipping 'bits_per_elem' distinct bits per index
// Returns (new_vector, list_of_infos) where info = (idx, positions)
static pair<vector<cpp_int>, vector<pair<size_t, vector<size_t>>>>
inject_multi(const vector<cpp_int> &values, int elem_count, int bits_per_elem, std::mt19937_64 &rng)
{
    if(values.empty() || elem_count<=0){
        return {values, {}};
    }
    int n = (int)values.size();
    elem_count = min(elem_count, n);

    vector<int> idxs(n);
    iota(idxs.begin(), idxs.end(), 0);
    shuffle(idxs.begin(), idxs.end(), rng);
    idxs.resize(elem_count);

    vector<cpp_int> new_vals = values;
    vector<pair<size_t, vector<size_t>>> infos;
    infos.reserve(elem_count);

    for(int idx: idxs){
        cpp_int v = new_vals[idx];
        size_t bl = max<size_t>(1, bitlen(v));  // ensure at least 1-bit space
        size_t k = min<size_t>(bits_per_elem, bl);

        // choose k distinct positions in [0, bl-1]
        // sample without replacement
        vector<size_t> positions;
        positions.reserve(k);
        // For large bl, sample by rejection
        unordered_set<size_t> used;
        while(positions.size() < k){
            size_t pos = (size_t)(rng() % bl);
            if(used.insert(pos).second) positions.push_back(pos);
        }

        cpp_int before = v;
        cpp_int after  = flip_bits(v, positions);
        new_vals[idx]  = after;
        infos.push_back({(size_t)idx, positions});
    }
    return {std::move(new_vals), std::move(infos)};
}

// Correct Barrett reduction for a vector
static vector<cpp_int> barrett_reduce_vector_correct(
    const vector<cpp_int> &X, const cpp_int &n, const cpp_int &mu, int k)
{
    vector<cpp_int> out;
    out.reserve(X.size());
    for(const cpp_int &x : X){
        cpp_int t = x * mu;          // big
        cpp_int s = t >> (2*k);      // floor(t / 2^(2k))
        cpp_int c = x - s * n;
        if(c < 0) c += n;
        else if(c >= n) c -= n;
        out.push_back(c);
    }
    return out;
}

// Faulty Barrett with detectors and 2D injection
static tuple<vector<cpp_int>, bool>
barrett_reduce_vector_faulty_with_checks(
    const vector<cpp_int> &X, const cpp_int &n, const cpp_int &mu, int k, int width,
    bool use_t_check, bool use_sn_check, bool use_final_check,
    int elem_count, int bits_per_elem, std::mt19937_64 &rng)
{
    // T-stage
    vector<cpp_int> t_list; t_list.reserve(X.size());
    for(const cpp_int &x: X) t_list.push_back(x * mu);
    cpp_int ecc_t_before = use_t_check ? fold_mod(accumulate(t_list.begin(), t_list.end(), cpp_int(0)), width) : 0;

    // s and SN
    vector<cpp_int> s_list; s_list.reserve(X.size());
    for(const cpp_int &t: t_list) s_list.push_back(t >> (2*k));

    vector<cpp_int> sn_list; sn_list.reserve(X.size());
    for(const cpp_int &s: s_list) sn_list.push_back(s * n);
    cpp_int ecc_sn_before = use_sn_check ? fold_mod(accumulate(sn_list.begin(), sn_list.end(), cpp_int(0)), width) : 0;

    bool inject_T = (rng() & 1ULL) != 0; // 50/50

    // apply faults
    if(inject_T){
        auto pair_fault = inject_multi(t_list, elem_count, bits_per_elem, rng);
        t_list = std::move(pair_fault.first);
        // refresh downstream
        s_list.clear(); s_list.reserve(X.size());
        for(const cpp_int &t: t_list) s_list.push_back(t >> (2*k));
        sn_list.clear(); sn_list.reserve(X.size());
        for(const cpp_int &s: s_list) sn_list.push_back(s * n);
    }else{
        auto pair_fault = inject_multi(sn_list, elem_count, bits_per_elem, rng);
        sn_list = std::move(pair_fault.first);
    }

    // detections
    bool detect_t = false, detect_sn = false;
    if(use_t_check && inject_T){
        cpp_int ecc_t_after = fold_mod(accumulate(t_list.begin(), t_list.end(), cpp_int(0)), width);
        detect_t = (ecc_t_after != ecc_t_before);
    }
    if(use_sn_check){
        cpp_int ecc_sn_after = fold_mod(accumulate(sn_list.begin(), sn_list.end(), cpp_int(0)), width);
        detect_sn = (ecc_sn_after != ecc_sn_before);
    }

    // final stage and final flags
    vector<cpp_int> c_list; c_list.reserve(X.size());
    bool detect_final = false;
    for(size_t i=0;i<X.size();++i){
        cpp_int c = X[i] - sn_list[i];
        bool flag = false;
        if(use_final_check){
            flag = !(cpp_int(0) <= c && c < (n<<1)); // 0 <= c < 2n
        }
        if(c < 0) c += n;
        else if(c >= n) c -= n;
        c_list.push_back(c);
        detect_final = detect_final || flag;
    }

    bool detected_any = detect_t || detect_sn || (use_final_check && detect_final);
    return {std::move(c_list), detected_any};
}

// -------------------- Main Monte Carlo per (fw, scheme) --------------------
struct Counts { uint64_t TP=0, FP=0, TN=0, FN=0; };

static Counts run_experiment(int fold_width, const Scheme &sch){
    // shared prime for this (fw, scheme)
    std::random_device rd;
    std::mt19937_64 seeder(rd());
    uint64_t p_u64 = gen_prime_bitwidth(seeder, BITWIDTH_PRIME);
    cpp_int p = p_u64;

    int k = (int)bitlen(p);
    cpp_int mu = (cpp_int(1) << (2*k)) / p;  // floor(2^(2k)/p)

    Counts total;

    omp_set_num_threads(16);

    #pragma omp parallel
    {
        // thread-local RNG
        std::mt19937_64 rng(rd() ^ ((uint64_t)omp_get_thread_num()*0x9e3779b97f4a7c15ULL));

        uint64_t TP=0, FP=0, TN=0, FN=0;

        #pragma omp for schedule(static)
        for(int trial=0; trial<TRIALS; ++trial){
            // Build random vectors A,B in [1, p-1]
            vector<cpp_int> X; X.reserve(VECTOR_LEN);
            for(size_t i=0;i<VECTOR_LEN;++i){
                uint64_t a = 1 + (rng() % (p_u64 - 1));
                uint64_t b = 1 + (rng() % (p_u64 - 1));
                cpp_int ai = a, bi = b;
                X.push_back(ai * bi);
            }

            // Correct output
            vector<cpp_int> out_ok = barrett_reduce_vector_correct(X, p, mu, k);

            // Faulty with detectors
            auto tup = barrett_reduce_vector_faulty_with_checks(
                X, p, mu, k, fold_width,
                sch.use_t, sch.use_sn, sch.use_final,
                INJECT_ELEM_COUNT, BITFLIPS_PER_ELEM, rng
            );
            const vector<cpp_int> &out_fault = get<0>(tup);
            bool detected_any = get<1>(tup);

            bool harmful = (out_fault != out_ok);
            if(harmful){
                if(detected_any) ++TP;
                else             ++FN;
            }else{
                if(detected_any) ++FP;
                else             ++TN;
            }
        }

        #pragma omp atomic
        total.TP += TP;
        #pragma omp atomic
        total.FP += FP;
        #pragma omp atomic
        total.TN += TN;
        #pragma omp atomic
        total.FN += FN;
    }

    return total;
}

// -------------------- Driver --------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    cout << "FOLD_WIDTH,USE_T,USE_SN,USE_FINAL,DetectionRate,TP,FP,TN,FN\n";

    for(int fw: FOLD_WIDTHS){
        for(const auto &sch: SCHEMES){
            Counts c = run_experiment(fw, sch);
            uint64_t harmful = c.TP + c.FN;
            double detection_rate = harmful ? (double)c.TP / (double)harmful : 1.0;
            cout << fw << ","
                 << (int)sch.use_t << ","
                 << (int)sch.use_sn << ","
                 << (int)sch.use_final << ","
                 << fixed << setprecision(4) << detection_rate << ","
                 << c.TP << "," << c.FP << "," << c.TN << "," << c.FN << "\n";
        }
    }
    return 0;
}
