// internal_injection_foldstudy_omp.cpp
// Build: g++ -O3 -std=gnu++17 -fopenmp internal_injection_foldstudy_omp.cpp -o foldstudy
// Run:   ./foldstudy --mode SBF --trials 1000 --vec-len 8192 --folds 1,2,4,8,16,32,64 --seed 1
//
// Fault model:
// - Inject only at multiplication outputs: mul1 = t*mu, mul2 = s*q.
// - Errors touch only low-K bits (< q bitwidth). MOF randomizes those K-bit signatures.
// - No range checks. Faults act in the signature domain; we do not propagate into arithmetic.
//
// Modes: SBF, DBF, SBF+SBF, SBF+DBF, MOF1, MOF2, MOF+SBF, MOF+DBF
// Protections:
//   instra: fold-level XOR over (lowK(mul1) XOR lowK(mul2))
//   sum   : fold-level XOR over lowK(c)  [no propagation here → near-zero detection]

#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

using u64 = uint64_t;
using u128 = unsigned __int128;

// ---------- deterministic Miller-Rabin for 64-bit ----------
static u64 mulmod_u128(u64 a, u64 b, u64 m) {
    return (u128)a * (u128)b % m;
}
static u64 powmod_u128(u64 a, u64 e, u64 m) {
    u64 r = 1 % m;
    while (e) {
        if (e & 1) r = mulmod_u128(r, a, m);
        a = mulmod_u128(a, a, m);
        e >>= 1;
    }
    return r;
}
static bool is_probable_prime_u64(u64 n) {
    if (n < 2) return false;
    for (u64 p : {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull,31ull,37ull}) {
        if (n % p == 0) return n == p;
    }
    u64 d = n - 1, r = 0;
    while ((d & 1) == 0) { d >>= 1; ++r; }
    // Deterministic set for 64-bit
    for (u64 a : {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull}) {
        if (a % n == 0) continue;
        u64 x = powmod_u128(a % n, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (u64 i = 1; i < r; ++i) {
            x = mulmod_u128(x, x, n);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}
static u64 next_prime_ge(u64 n) {
    if (n <= 2) return 2;
    if ((n & 1) == 0) ++n;
    while (!is_probable_prime_u64(n)) n += 2;
    return n;
}

// ---------- global params ----------
static const int BITWIDTH_PRIME = 37;
static const int DEFAULT_VEC_LEN = 8192;

// ---------- Barrett params ----------
struct BarrettCtx {
    int K;        // ceil(log2 q)
    u128 mu;      // floor(2^(2K)/q)
};
static BarrettCtx make_barrett_ctx(u64 q) {
    BarrettCtx c;
    c.K = 64 - __builtin_clzll(q - 1); // bit_length(q-1)
    u128 one = (u128)1;
    u128 num = (one << (2 * c.K));     // 2^(2K) (fits in 128 since K<=63 here; we use 37)
    c.mu = num / (u128)q;
    return c;
}

// ---------- RNG helpers ----------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
static mt19937_64 make_rng(uint64_t seed) {
    return mt19937_64(splitmix64(seed));
}

// ---------- bit operations within low K bits ----------
static inline u64 maskK_bits(int K) {
    return (K >= 64) ? ~0ull : ((1ull << K) - 1ull);
}
static inline u64 flip_k_limited(u64 x, mt19937_64 &rng, int k, int K) {
    // Flip k distinct bits in [0, K)
    k = min(k, max(1, K));
    if (k == 1) {
        uniform_int_distribution<int> dist(0, max(0, K - 1));
        int b = dist(rng);
        return x ^ (1ull << b);
    } else { // k==2
        if (K == 1) return x ^ 1ull; // only bit 0 exists
        uniform_int_distribution<int> dist(0, K - 1);
        int b1 = dist(rng), b2 = dist(rng);
        while (b2 == b1) b2 = dist(rng);
        x ^= (1ull << b1);
        x ^= (1ull << b2);
        return x;
    }
}
static inline u64 rand_sig(mt19937_64 &rng, int K) {
    if (K <= 0) return 0;
    u64 r = rng();
    return r & maskK_bits(K);
}

// ---------- one Barrett reduction; return c, lowK(mul1), lowK(mul2) ----------
static inline void barrett_reduce_signatures(u64 ti, u64 q, const BarrettCtx &bc,
                                             u64 &c, u64 &s1, u64 &s2) {
    const int K = bc.K;
    const u64 maskK = maskK_bits(K);

    // mul1 = ti * mu (use 128-bit)
    u128 mul1 = (u128)ti * bc.mu;
    s1 = (u64)(mul1 & maskK);

    // si = floor(mul1 / 2^(2K))
    u128 si = (mul1 >> (2 * K));

    // mul2 = si * q
    u128 mul2 = si * (u128)q;
    s2 = (u64)(mul2 & maskK);

    // ci = ti - mul2; then conditional subtract once
    u128 ci128 = (u128)ti - mul2;
    u64 ci = (u64)ci128;
    if (ci >= q) ci -= q;
    c = ci;
}

// ---------- injection planning ----------
enum class Mode {
    SBF, DBF, SBF_SBF, SBF_DBF, MOF1, MOF2, MOF_SBF, MOF_DBF
};
static Mode parse_mode(const string &m) {
    if (m == "SBF") return Mode::SBF;
    if (m == "DBF") return Mode::DBF;
    if (m == "SBF+SBF") return Mode::SBF_SBF;
    if (m == "SBF+DBF") return Mode::SBF_DBF;
    if (m == "MOF1") return Mode::MOF1;
    if (m == "MOF2") return Mode::MOF2;
    if (m == "MOF+SBF") return Mode::MOF_SBF;
    if (m == "MOF+DBF") return Mode::MOF_DBF;
    throw runtime_error("Unknown mode");
}

static inline void apply_sbf_or_dbf_to_signatures(u64 &s1, u64 &s2, mt19937_64 &rng, int K, bool dbf) {
    bool choose_mul1 = (rng() & 1ull);
    if (!dbf) { // SBF
        if (choose_mul1) s1 = flip_k_limited(s1, rng, 1, K);
        else             s2 = flip_k_limited(s2, rng, 1, K);
    } else {    // DBF
        if (choose_mul1) s1 = flip_k_limited(s1, rng, 2, K);
        else             s2 = flip_k_limited(s2, rng, 2, K);
    }
}
static inline void apply_mof_to_signatures(u64 &s1, u64 &s2, mt19937_64 &rng, int K) {
    s1 = rand_sig(rng, K);
    s2 = rand_sig(rng, K);
}

// ---------- one trial over a vector; return detected? ----------
static bool simulate_once_fold(u64 q, const BarrettCtx &bc, int n, int fold,
                               mt19937_64 &rng, Mode mode, bool protect_instra) {
    const int K = bc.K;
    const u64 maskK = maskK_bits(K);
    bool detected_any = false;

    // plan indices
    int i1 = uniform_int_distribution<int>(0, n - 1)(rng);
    int i2 = i1;
    auto pick_two = [&](int &a, int &b){
        a = uniform_int_distribution<int>(0, n - 1)(rng);
        do { b = uniform_int_distribution<int>(0, n - 1)(rng); } while (b == a);
    };

    vector<char> is_mof(n, 0);
    vector<char> inj_kind(n, 0); // 0:none, 1:SBF, 2:DBF

    switch (mode) {
        case Mode::SBF:
            inj_kind[i1] = 1; break;
        case Mode::DBF:
            inj_kind[i1] = 2; break;
        case Mode::SBF_SBF: {
            pick_two(i1, i2);
            inj_kind[i1] = 1; inj_kind[i2] = 1; break;
        }
        case Mode::SBF_DBF: {
            pick_two(i1, i2);
            if (rng() & 1ull) { inj_kind[i1] = 1; inj_kind[i2] = 2; }
            else               { inj_kind[i1] = 2; inj_kind[i2] = 1; }
            break;
        }
        case Mode::MOF1:
            is_mof[i1] = 1; break;
        case Mode::MOF2:
            pick_two(i1, i2);
            is_mof[i1] = 1; is_mof[i2] = 1; break;
        case Mode::MOF_SBF: {
            pick_two(i1, i2);
            is_mof[i1] = 1; inj_kind[i2] = 1; break;
        }
        case Mode::MOF_DBF: {
            pick_two(i1, i2);
            is_mof[i1] = 1; inj_kind[i2] = 2; break;
        }
    }

    u64 acc_g = 0, acc_f = 0;

    for (int idx = 0; idx < n; ++idx) {
        // random operands
        u64 a = uniform_int_distribution<u64>(0, q - 1)(rng);
        u64 b = uniform_int_distribution<u64>(0, q - 1)(rng);
        u64 ti = a * b;

        u64 c_g, s1_g, s2_g;
        barrett_reduce_signatures(ti, q, bc, c_g, s1_g, s2_g);

        u64 s1_f = s1_g, s2_f = s2_g;
        u64 c_f  = c_g; // no propagation

        if (is_mof[idx]) {
            apply_mof_to_signatures(s1_f, s2_f, rng, K);
        }
        if (inj_kind[idx] == 1) {
            apply_sbf_or_dbf_to_signatures(s1_f, s2_f, rng, K, /*dbf=*/false);
        } else if (inj_kind[idx] == 2) {
            apply_sbf_or_dbf_to_signatures(s1_f, s2_f, rng, K, /*dbf=*/true);
        }

        u64 sig_g, sig_f;
        if (protect_instra) {
            sig_g = (s1_g ^ s2_g) & maskK;
            sig_f = (s1_f ^ s2_f) & maskK;
        } else {
            sig_g = c_g & maskK;
            sig_f = c_f & maskK; // unchanged under signature-only fault model
        }

        acc_g ^= sig_g;
        acc_f ^= sig_f;

        if (((idx + 1) % fold) == 0) {
            if (acc_g != acc_f) detected_any = true;
            acc_g = 0; acc_f = 0;
        }
    }
    if ((n % fold) != 0) {
        if (acc_g != acc_f) detected_any = true;
    }
    return detected_any;
}

// ---------- main sweep ----------
int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // defaults
    int trials = 200;
    uint64_t seed = 1;
    int n = DEFAULT_VEC_LEN;
    string folds_s = "1,2,4,8,16,32,64";
    string mode_s = "SBF";
    int threads = max(1, omp_get_max_threads());

    // parse CLI
    for (int i = 1; i < argc; ++i) {
        string k = argv[i];
        auto need = [&](const char* err){ if (i + 1 >= argc) { cerr << err << "\n"; exit(1);} return string(argv[++i]); };
        if (k == string("--trials")) trials = stoi(need("missing --trials value"));
        else if (k == string("--seed")) seed = stoull(need("missing --seed value"));
        else if (k == string("--vec-len")) n = stoi(need("missing --vec-len value"));
        else if (k == string("--folds")) folds_s = need("missing --folds value");
        else if (k == string("--mode")) mode_s = need("missing --mode value");
        else if (k == string("--threads")) threads = stoi(need("missing --threads value"));
        else if (k == string("--help") || k == string("-h")) {
            cout <<
"Usage: ./foldstudy [--trials N] [--seed S] [--vec-len N] [--folds list] [--mode M] [--threads T]\n"
"Modes: SBF | DBF | SBF+SBF | SBF+DBF | MOF1 | MOF2 | MOF+SBF | MOF+DBF\n";
            return 0;
        }
    }
    omp_set_num_threads(max(1, threads));

    // modulus q and Barrett ctx
    u64 Q = next_prime_ge(1ull << (BITWIDTH_PRIME - 1));
    auto bc = make_barrett_ctx(Q);

    // parse folds
    vector<int> folds;
    {
        string tmp = folds_s; replace(tmp.begin(), tmp.end(), ',', ' ');
        stringstream ss(tmp); int x;
        while (ss >> x) if (x > 0) folds.push_back(x);
        if (folds.empty()) { cerr << "No valid folds\n"; return 1; }
    }
    Mode mode = parse_mode(mode_s);

    cout << "q=" << Q << " (bits=" << (64 - __builtin_clzll(Q)) << "), "
         << "N=" << n << ", K=" << bc.K << "\n";
    cout << "Injection at mul outputs within <K bits; MOF randomizes signatures | mode=" << mode_s << "\n";
    cout << "Fold sweep: "; for (size_t i=0;i<folds.size();++i){ if(i) cout<<","; cout<<folds[i]; } cout << "\n";
    cout << "Strategy,Fold,Trials,Detected,Missed,MissRate\n";

    // protections: 0 -> instra, 1 -> sum
    for (int prot = 0; prot < 2; ++prot) {
        string prot_name = (prot == 0 ? "instra" : "sum");
        for (int fold : folds) {
            long long detected = 0;

            #pragma omp parallel for reduction(+:detected) schedule(static)
            for (int t = 0; t < trials; ++t) {
                // per-trial RNG: mix seed with trial/fold/protection
                uint64_t s = seed ^ (uint64_t)fold * 0x9e3779b97f4a7c15ull
                                   ^ (uint64_t)prot * 0xbf58476d1ce4e5b9ull
                                   ^ (uint64_t)t;
                auto rng = make_rng(s);
                bool det = simulate_once_fold(Q, bc, n, fold, rng, mode, /*protect_instra=*/prot==0);
                detected += det ? 1 : 0;
            }

            long long missed = trials - detected;
            double miss_rate = trials ? (double)missed / (double)trials : 0.0;
            cout << prot_name << "," << fold << "," << trials << ","
                 << detected << "," << missed << "," << fixed << setprecision(6) << miss_rate << "\n";
        }
    }
    return 0;
}
