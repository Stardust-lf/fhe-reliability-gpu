// sim_faults.cpp
// Build: g++ -O3 -std=c++17 -fopenmp sim_faults.cpp -o sim_faults
// Run  : ./sim_faults --n 1024 --qbits 16 --trials 100000 --fold_min 2 --fold_max 32 --fold_step 2 --out result.csv
// Notes: All comments in English per your requirement.

#include <bits/stdc++.h>
#ifdef _OPENMP
  #include <omp.h>
#endif

using namespace std;

// ---------- rng ----------
struct ThreadRNG {
    std::mt19937_64 eng;
    ThreadRNG(uint64_t seed, uint64_t tid) { eng.seed(seed ^ (0x9E3779B97F4A7C15ULL + (tid<<1))); }
    uint64_t urand64() { return eng(); }
    uint32_t urand32() { return static_cast<uint32_t>(eng()); }
    size_t randint(size_t lo, size_t hi) { // [lo, hi]
        std::uniform_int_distribution<size_t> d(lo, hi);
        return d(eng);
    }
    int bit_in_range(int qbits) {
        std::uniform_int_distribution<int> d(0, qbits-1);
        return d(eng);
    }
};

// ---------- fold mod (2^k + 1) ----------
static inline uint64_t mod_pos(int64_t x, uint64_t M) {
    int64_t r = x % (int64_t)M;
    if (r < 0) r += (int64_t)M;
    return (uint64_t)r;
}
static inline uint64_t fold_mod_2k_plus_1_u64(uint64_t x, int k) {
    const uint64_t M = (1ULL<<k) + 1ULL;
    const uint64_t mask = (k==64)?~0ULL:((1ULL<<k)-1ULL);
    // process 64 bits in chunks of k with alternating signs
    uint64_t acc = 0;
    int sign = 1;
    const int segs = (64 + k - 1)/k;
    for (int i=0;i<segs;i++){
        uint64_t seg = (x >> (i*k)) & mask;
        int64_t term = sign>0 ? (int64_t)seg : -(int64_t)seg;
        acc = (acc + mod_pos(term, M)) % M;
        sign = -sign;
    }
    return acc % M;
}

// ---------- modes ----------
enum Mode {
    SBF, DBF, SBF_SBF, SBF_DBF, MOF1, MOF2, MOF_SBF, MOF_DBF
};
static inline string mode_name(Mode m){
    switch(m){
        case SBF: return "SBF";
        case DBF: return "DBF";
        case SBF_SBF: return "SBF+SBF";
        case SBF_DBF: return "SBF+DBF";
        case MOF1: return "MOF1";
        case MOF2: return "MOF2";
        case MOF_SBF: return "MOF+SBF";
        case MOF_DBF: return "MOF+DBF";
    }
    return "?";
}
vector<Mode> parse_modes(string s){
    if (s=="all" || s.empty()){
        return {SBF,DBF,SBF_SBF,SBF_DBF,MOF1,MOF2,MOF_SBF,MOF_DBF};
    }
    vector<Mode> out;
    string token; stringstream ss(s);
    while(getline(ss, token, ',')){
        if(token=="SBF") out.push_back(SBF);
        else if(token=="DBF") out.push_back(DBF);
        else if(token=="SBF+SBF") out.push_back(SBF_SBF);
        else if(token=="SBF+DBF") out.push_back(SBF_DBF);
        else if(token=="MOF1") out.push_back(MOF1);
        else if(token=="MOF2") out.push_back(MOF2);
        else if(token=="MOF+SBF") out.push_back(MOF_SBF);
        else if(token=="MOF+DBF") out.push_back(MOF_DBF);
    }
    return out;
}

// ---------- helpers ----------
static inline uint64_t apply_rand_low(uint64_t v, int qbits, ThreadRNG& R){
    uint64_t mask = (qbits==64)?~0ULL:((1ULL<<qbits)-1ULL);
    uint64_t new_low = (qbits==64)? R.urand64() : (R.urand64() & mask);
    return (v & ~mask) | (new_low & mask);
}
static inline int64_t flip_one_delta(uint64_t old, int bit){
    uint64_t bm = (1ULL<<bit);
    return ( (old & bm) ? -(int64_t)bm : (int64_t)bm );
}
static inline uint64_t add_u64_delta(uint64_t base, int64_t delta){
    if(delta>=0) return base + (uint64_t)delta;
    else return base - (uint64_t)(-delta);
}

// ---------- injection: build deltas per index ----------
struct DeltaSet {
    // idx -> signed delta
    vector<pair<size_t,int64_t>> vec; // small (<=2 entries)
};

DeltaSet inject_and_delta(const vector<uint64_t>& c, int qbits, Mode mode, ThreadRNG& R){
    const size_t n = c.size();
    auto pick_two = [&](size_t& i1, size_t& i2){
        i1 = R.randint(0,n-1);
        do { i2 = R.randint(0,n-1); } while(i2==i1);
    };

    unordered_map<size_t, vector<int>> ops; // int>=0 bit; -1 means RAND
    if(mode==SBF){
        size_t i = R.randint(0,n-1);
        ops[i].push_back(R.bit_in_range(qbits));
    } else if(mode==DBF){
        size_t i = R.randint(0,n-1);
        int b1 = R.bit_in_range(qbits), b2;
        do { b2 = R.bit_in_range(qbits); } while(b2==b1);
        ops[i].push_back(b1); ops[i].push_back(b2);
    } else if(mode==SBF_SBF){
        size_t i1,i2; pick_two(i1,i2);
        ops[i1].push_back(R.bit_in_range(qbits));
        ops[i2].push_back(R.bit_in_range(qbits));
    } else if(mode==SBF_DBF){
        size_t i1,i2; pick_two(i1,i2);
        ops[i1].push_back(R.bit_in_range(qbits));
        int b2 = R.bit_in_range(qbits), b3;
        do { b3 = R.bit_in_range(qbits); } while(b3==b2);
        ops[i2].push_back(b2); ops[i2].push_back(b3);
    } else if(mode==MOF1){
        size_t i = R.randint(0,n-1);
        ops[i].push_back(-1);
    } else if(mode==MOF2){
        size_t i1,i2; pick_two(i1,i2);
        ops[i1].push_back(-1);
        ops[i2].push_back(-1);
    } else if(mode==MOF_SBF){
        size_t i1,i2; pick_two(i1,i2);
        ops[i1].push_back(-1);
        ops[i2].push_back(R.bit_in_range(qbits));
    } else if(mode==MOF_DBF){
        size_t i1,i2; pick_two(i1,i2);
        ops[i1].push_back(-1);
        int b2 = R.bit_in_range(qbits), b3;
        do { b3 = R.bit_in_range(qbits); } while(b3==b2);
        ops[i2].push_back(b2); ops[i2].push_back(b3);
    }

    DeltaSet ds;
    ds.vec.reserve(2);
    const uint64_t lowmask = (qbits==64)?~0ULL:((1ULL<<qbits)-1ULL);
    for(auto& kv: ops){
        size_t idx = kv.first;
        uint64_t cur = c[idx];
        int64_t delta = 0;
        for(int op: kv.second){
            if(op<0){ // RAND
                uint64_t old_low = cur & lowmask;
                uint64_t new_low = (qbits==64)? R.urand64() : (R.urand64() & lowmask);
                int64_t d = (int64_t)((uint64_t)new_low) - (int64_t)((uint64_t)old_low);
                delta += d;
                cur = (cur & ~lowmask) | (new_low & lowmask);
            } else { // bit flip
                delta += flip_one_delta(cur, op);
                cur ^= (1ULL<<op);
            }
        }
        ds.vec.emplace_back(idx, delta);
    }
    return ds;
}

// ---------- main experiment ----------
struct Args {
    size_t n = 1024;
    int qbits = 37;
    int fold_min = 2, fold_max = 32, fold_step = 2;
    uint64_t trials = 100000000;
    uint64_t seed = 1;
    string modes = "all";
    string out = ""; // empty => stdout
};

Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;i++){
        string k = argv[i];
        auto next = [&](int i){ if(i+1>=argc) { fprintf(stderr,"missing value for %s\n",k.c_str()); exit(1);} return string(argv[i+1]); };
        if(k=="--n") a.n = stoull(next(i++));
        else if(k=="--qbits") a.qbits = stoi(next(i++));
        else if(k=="--trials") a.trials = stoull(next(i++));
        else if(k=="--fold_min") a.fold_min = stoi(next(i++));
        else if(k=="--fold_max") a.fold_max = stoi(next(i++));
        else if(k=="--fold_step") a.fold_step = stoi(next(i++));
        else if(k=="--seed") a.seed = stoull(next(i++));
        else if(k=="--modes") a.modes = next(i++);
        else if(k=="--out") a.out = next(i++);
    }
    return a;
}

int main(int argc, char** argv){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    Args A = parse_args(argc, argv);
    auto modes = parse_modes(A.modes);

    // generate base c
    std::mt19937_64 g(A.seed);
    vector<uint64_t> c(A.n);
    for(size_t i=0;i<A.n;i++) c[i] = g();

    // precompute base sums per k
    vector<int> ks;
    for(int k=A.fold_min; k<=A.fold_max; k+=A.fold_step) ks.push_back(k);
    unordered_map<int, uint64_t> base_sum;
    for(int k: ks){
        uint64_t M = (1ULL<<k)+1ULL;
        uint64_t s = 0;
        for(size_t i=0;i<A.n;i++){
            s = (s + fold_mod_2k_plus_1_u64(c[i], k)) % M;
        }
        base_sum[k] = s;
    }

    // open output
    unique_ptr<ofstream> fout;
    ostream* out = &cout;
    if(!A.out.empty()){
        fout.reset(new ofstream(A.out));
        out = fout.get();
    }
    (*out) << "strategy,mode,fold,miss_rate\n";

    // loop k, mode
    for(int k: ks){
        uint64_t M = (1ULL<<k)+1ULL;
        for(auto m: modes){
            // miss counters
            uint64_t miss_intra = 0, miss_inter = 0;

            #pragma omp parallel
            {
                ThreadRNG R(A.seed + 1337, 
                #ifdef _OPENMP
                    omp_get_thread_num()
                #else
                    0
                #endif
                );
                uint64_t loc_mi = 0, loc_mj = 0;

                #pragma omp for
                for(long long t=0; t<(long long)A.trials; t++){
                    // inject at most 2 indices
                    DeltaSet ds = inject_and_delta(c, A.qbits, m, R);

                    // intra: compare fold(c[idx]) vs fold(c[idx]+delta)
                    bool detected_intra = false;
                    for(auto &p: ds.vec){
                        size_t idx = p.first;
                        int64_t delta = p.second;
                        uint64_t before = fold_mod_2k_plus_1_u64(c[idx], k);
                        uint64_t after  = fold_mod_2k_plus_1_u64(add_u64_delta(c[idx], delta), k);
                        if(before != after){ detected_intra = true; break; }
                    }

                    // inter: update base sum only for changed indices
                    uint64_t sum_faulty = base_sum[k];
                    for(auto &p: ds.vec){
                        size_t idx = p.first;
                        int64_t delta = p.second;
                        uint64_t before = fold_mod_2k_plus_1_u64(c[idx], k);
                        uint64_t after  = fold_mod_2k_plus_1_u64(add_u64_delta(c[idx], delta), k);
                        // sum_faulty = (sum_faulty - before + after) mod M
                        int64_t tmp = (int64_t)sum_faulty - (int64_t)before + (int64_t)after;
                        sum_faulty = (uint64_t)mod_pos(tmp, M);
                    }
                    bool detected_inter = (sum_faulty != base_sum[k]);

                    if(!detected_intra) loc_mi++;
                    if(!detected_inter) loc_mj++;
                }

                #pragma omp atomic
                miss_intra += loc_mi;
                #pragma omp atomic
                miss_inter += loc_mj;
            }

            double mr_intra = (double)miss_intra / (double)A.trials;
            double mr_inter = (double)miss_inter / (double)A.trials;

            (*out) << "intra," << mode_name(m) << "," << k << "," << std::setprecision(17) << mr_intra << "\n";
            (*out) << "inter," << mode_name(m) << "," << k << "," << std::setprecision(17) << mr_inter << "\n";
        }
    }

    if(fout) fout->close();
    return 0;
}
