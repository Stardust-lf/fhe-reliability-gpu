// sim_faults_qbits.cpp
// Build: g++ -O3 -std=c++17 -fopenmp sim_faults_qbits.cpp -o sim_faults_qbits
// Run  : ./sim_faults_qbits --n 1024 --fold_fixed 24 --trials 100000 --q_min 10 --q_max 20 --q_step 1 --out result_qbits.csv
// Notes: Loop over qbits (system prime width) with fixed fold width.

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
    SBF, DBF, SBF_SBF, SBF_DBF, MOF, MOF_MOF, MOF_SBF, MOF_DBF
};
static inline string mode_name(Mode m){
    switch(m){
        case SBF: return "SBF";
        case DBF: return "DBF";
        case SBF_SBF: return "SBF+SBF";
        case SBF_DBF: return "SBF+DBF";
        case MOF: return "MOF";
        case MOF_MOF: return "MOF+MOF";
        case MOF_SBF: return "MOF+SBF";
        case MOF_DBF: return "MOF+DBF";
    }
    return "?";
}

// Map strings to modes (Updated to match your previous CSV output style)
vector<Mode> parse_modes(string s){
    if (s=="all" || s.empty()){
        return {SBF,DBF,SBF_SBF,SBF_DBF,MOF,MOF_MOF,MOF_SBF,MOF_DBF};
    }
    vector<Mode> out;
    string token; stringstream ss(s);
    while(getline(ss, token, ',')){
        if(token=="SBF") out.push_back(SBF);
        else if(token=="DBF") out.push_back(DBF);
        else if(token=="SBF+SBF") out.push_back(SBF_SBF);
        else if(token=="SBF+DBF") out.push_back(SBF_DBF);
        else if(token=="MOF") out.push_back(MOF);
        else if(token=="MOF+MOF") out.push_back(MOF_MOF);
        else if(token=="MOF+SBF") out.push_back(MOF_SBF);
        else if(token=="MOF+DBF") out.push_back(MOF_DBF);
    }
    return out;
}

// ---------- helpers ----------
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
    vector<pair<size_t,int64_t>> vec; 
};

// Note: 'c' passed here must already be masked to 'qbits' if we want strictly valid inputs,
// but the injection logic handles bit boundaries via 'qbits' param.
DeltaSet inject_and_delta(const vector<uint64_t>& c, int qbits, Mode mode, ThreadRNG& R){
    const size_t n = c.size();
    auto pick_two = [&](size_t& i1, size_t& i2){
        i1 = R.randint(0,n-1);
        do { i2 = R.randint(0,n-1); } while(i2==i1);
    };

    unordered_map<size_t, vector<int>> ops; // int>=0 bit; -1 means RAND
    
    // Logic mapping to modes
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
    } else if(mode==MOF){
        size_t i = R.randint(0,n-1);
        ops[i].push_back(-1);
    } else if(mode==MOF_MOF){
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
            if(op<0){ // RAND (Modular Overflow / Random value change)
                uint64_t old_low = cur & lowmask;
                uint64_t new_low = (qbits==64)? R.urand64() : (R.urand64() & lowmask);
                // Calculate arithmetic difference
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
    int fold_fixed = 24;      // Fixed fold width
    int q_min = 10;           // Start qbits
    int q_max = 20;           // End qbits
    int q_step = 1;           // Step
    uint64_t trials = 100000000;
    uint64_t seed = 1;
    string modes = "all";
    string out = "";
};

Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;i++){
        string k = argv[i];
        auto next = [&](int i){ if(i+1>=argc) { fprintf(stderr,"missing value for %s\n",k.c_str()); exit(1);} return string(argv[i+1]); };
        if(k=="--n") a.n = stoull(next(i++));
        else if(k=="--fold_fixed") a.fold_fixed = stoi(next(i++));
        else if(k=="--q_min") a.q_min = stoi(next(i++));
        else if(k=="--q_max") a.q_max = stoi(next(i++));
        else if(k=="--q_step") a.q_step = stoi(next(i++));
        else if(k=="--trials") a.trials = stoull(next(i++));
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

    // generate raw random data (64-bit)
    std::mt19937_64 g(A.seed);
    vector<uint64_t> c_raw(A.n);
    for(size_t i=0;i<A.n;i++) c_raw[i] = g();

    // open output
    unique_ptr<ofstream> fout;
    ostream* out = &cout;
    if(!A.out.empty()){
        fout.reset(new ofstream(A.out));
        out = fout.get();
    }
    // Header changed: fold -> qbits
    (*out) << "strategy,mode,qbits,miss_rate\n";

    // Loop over qbits (System Prime Size)
    vector<int> q_list;
    for(int q=A.q_min; q<=A.q_max; q+=A.q_step) q_list.push_back(q);

    for(int q : q_list){
        // 1. Prepare data for this qbit size
        //    Mask the raw data so it effectively becomes q-bit data
        uint64_t qmask = (q==64)? ~0ULL : ((1ULL<<q)-1ULL);
        vector<uint64_t> c(A.n);
        for(size_t i=0; i<A.n; i++) c[i] = c_raw[i] & qmask;

        // 2. Precompute base sum for fixed fold width
        //    Using the fixed fold k = A.fold_fixed
        int k = A.fold_fixed;
        uint64_t M = (1ULL<<k)+1ULL;
        uint64_t base_sum = 0;
        for(size_t i=0; i<A.n; i++){
            base_sum = (base_sum + fold_mod_2k_plus_1_u64(c[i], k)) % M;
        }

        for(auto m: modes){
            uint64_t miss_intra = 0, miss_inter = 0;

            #pragma omp parallel
            {
                ThreadRNG R(A.seed + q * 1024, // vary seed by qbits to avoid correlation
                #ifdef _OPENMP
                    omp_get_thread_num()
                #else
                    0
                #endif
                );
                uint64_t loc_mi = 0, loc_mj = 0;

                #pragma omp for
                for(long long t=0; t<(long long)A.trials; t++){
                    // inject faults assuming q-bit system
                    DeltaSet ds = inject_and_delta(c, q, m, R);

                    // intra check
                    bool detected_intra = false;
                    for(auto &p: ds.vec){
                        size_t idx = p.first;
                        int64_t delta = p.second;
                        uint64_t before = fold_mod_2k_plus_1_u64(c[idx], k);
                        uint64_t after  = fold_mod_2k_plus_1_u64(add_u64_delta(c[idx], delta), k);
                        if(before != after){ detected_intra = true; break; }
                    }

                    // inter check
                    uint64_t sum_faulty = base_sum;
                    for(auto &p: ds.vec){
                        size_t idx = p.first;
                        int64_t delta = p.second;
                        uint64_t before = fold_mod_2k_plus_1_u64(c[idx], k);
                        uint64_t after  = fold_mod_2k_plus_1_u64(add_u64_delta(c[idx], delta), k);
                        
                        // sum update
                        int64_t tmp = (int64_t)sum_faulty - (int64_t)before + (int64_t)after;
                        sum_faulty = (uint64_t)mod_pos(tmp, M);
                    }
                    bool detected_inter = (sum_faulty != base_sum);

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

            // Output qbits in the 3rd column
            (*out) << "intra," << mode_name(m) << "," << q << "," << std::setprecision(17) << mr_intra << "\n";
            (*out) << "inter," << mode_name(m) << "," << q << "," << std::setprecision(17) << mr_inter << "\n";
        }
        
        // Optional progress log
        // cerr << "Finished qbits=" << q << endl;
    }

    if(fout) fout->close();
    return 0;
}
