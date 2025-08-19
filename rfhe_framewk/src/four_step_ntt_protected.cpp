// mc_fhe.cpp
// FHE fault-injection Monte Carlo in C++17 with OpenMP.
// Mirrors the Python pipeline and fault models; parallel over trials.
// Build: g++ -O3 -march=native -fopenmp mc_fhe.cpp -o mc_fhe

#include <bits/stdc++.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using u64 = uint64_t;
using u128 = __uint128_t;

// -------- CLI parsing --------
struct Args {
    int pbits = 30;
    int W = 4;
    int N = 64;            // N = S*S; S must be sqrt(N)
    long long trials = 1000000;
    uint64_t seed = 42;
    uint64_t p_override = 0; // 0 => search
};

bool parse_int(const char* s, long long& out) {
    char* end=nullptr; errno=0;
    long long v = std::strtoll(s,&end,10);
    if(errno || end==s || *end!='\0') return false;
    out = v; return true;
}

Args parse_args(int argc, char** argv){
    Args a;
    for(int i=1;i<argc;i++){
        std::string k = argv[i];
        auto need = [&](int i){ return i+1<argc; };
        if(k=="--pbits" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.pbits=(int)v; }
        else if(k=="--W" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.W=(int)v; }
        else if(k=="--N" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.N=(int)v; }
        else if(k=="--trials" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.trials=v; }
        else if(k=="--seed" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.seed=(uint64_t)v; }
        else if(k=="--p" && need(i)){ long long v; if(parse_int(argv[++i],v)) a.p_override=(uint64_t)v; }
        else { /* ignore unknown */ }
    }
    return a;
}

// -------- RNG helpers (thread-local) --------
struct ThreadRNG {
    std::mt19937_64 eng;
    ThreadRNG(uint64_t s){ eng.seed(s); }
    uint64_t randint(uint64_t lo, uint64_t hi){ // inclusive
        std::uniform_int_distribution<u64> dist(lo,hi);
        return dist(eng);
    }
    int randi(int lo, int hi){ std::uniform_int_distribution<int> d(lo,hi); return d(eng); }
    bool coin(){ return (eng() & 1ull); }
    template<typename T>
    T choice(const std::vector<T>& vals){ std::uniform_int_distribution<size_t> d(0, vals.size()-1); return vals[d(eng)]; }
};

// -------- Miller-Rabin for 64-bit --------
u64 modmul(u64 a, u64 b, u64 m){ return (u128)a * b % m; }
u64 modpow(u64 a, u64 e, u64 m){
    u64 r=1; while(e){ if(e&1) r=modmul(r,a,m); a=modmul(a,a,m); e>>=1; } return r;
}
bool is_probable_prime(u64 n){
    if(n<2) return false;
    for(u64 p: {2ull,3ull,5ull,7ull,11ull,13ull,17ull,19ull,23ull,29ull}){
        if(n%p==0) return n==p;
    }
    u64 d = n-1, s=0;
    while((d&1)==0){ d>>=1; s++; }
    // Deterministic bases for 64-bit
    for(u64 a: {2ull,3ull,5ull,7ull,11ull,13ull,17ull}){
        if(a%n==0) continue;
        u64 x = modpow(a,d,n);
        if(x==1 || x==n-1) continue;
        bool pass=false;
        for(u64 r=1;r<s;r++){
            x = modmul(x,x,n);
            if(x==n-1){ pass=true; break; }
        }
        if(pass) continue;
        return false;
    }
    return true;
}
u64 find_prime_with_bitlen(int bits, ThreadRNG& rng){
    if(bits<2) throw std::runtime_error("bitlen must be >=2");
    while(true){
        u64 hi = 1ull<<(bits-1);
        u64 n = hi | (rng.randint(0, (1ull<<(bits-1))-1)) | 1ull;
        if(is_probable_prime(n)) return n;
    }
}

// -------- Math helpers --------
inline u64 addmod(u64 a,u64 b,u64 m){ u64 s=a+b; if(s>=m || s<a) s=(s%m + b%m)%m; return s%m; }
inline u64 mulmod(u64 a,u64 b,u64 m){ return (u128)a*b % m; }

u64 sum_mod(const std::vector<u64>& v, u64 m){
    u64 acc=0; for(u64 x: v){ acc += x; acc %= m; } return acc;
}
u64 dot_mod(const std::vector<u64>& a, const std::vector<u64>& b, u64 m){
    u64 acc=0; for(size_t i=0;i<a.size();i++){ acc = (acc + (u128)a[i]*b[i]%m)%m; } return acc;
}

// Matrix ops in row-major vectors
void matmul_mod(const std::vector<u64>& A, const std::vector<u64>& B, std::vector<u64>& C, int S, u64 P){
    // C = A @ B mod P
    for(int i=0;i<S;i++){
        for(int j=0;j<S;j++){
            u128 acc=0;
            for(int k=0;k<S;k++){
                acc += (u128)A[i*S+k]*B[k*S+j];
                if(acc > (u128)P*4) acc %= P;
            }
            C[i*S+j] = (u64)(acc % P);
        }
    }
}
bool matmul_with_protection(const std::vector<u64>& A, const std::vector<u64>& B, const std::vector<u64>& C, int S, u64 P){
    // sum(C) == col_sums(A)^T * row_sums(B) (mod P)
    std::vector<u64> col(S,0), row(S,0);
    // col sums of A (over rows)
    for(int j=0;j<S;j++){
        u64 s=0; for(int i=0;i<S;i++){ s += A[i*S+j]; s%=P; } col[j]=s;
    }
    // row sums of B
    for(int i=0;i<S;i++){
        u64 s=0; for(int j=0;j<S;j++){ s += B[i*S+j]; s%=P; } row[i]=s;
    }
    u64 lhs = dot_mod(col,row,P);
    u64 rhs = 0;
    for(int i=0;i<S*S;i++){ rhs += C[i]; rhs%=P; }
    return lhs==rhs;
}

u64 fold_mod(u64 x, int W, u64 M){
    // Folding checksum modulo (2^W - 1)
    const u64 mask = (W==64)?~0ull : ((1ull<<W)-1ull);
    u64 s=0;
    while(x){
        s += (x & mask);
        x >>= W;
    }
    return s % M;
}
bool elementwise_with_fold(const std::vector<u64>& X, const std::vector<u64>& T,
                           const std::vector<u64>& Y, int W, u64 M){
    // Check: sum(fold(X_i)*fold(T_i)) == sum(fold(Y_i)) (mod M)
    u64 Sin=0, Sout=0;
    for(size_t i=0;i<X.size();i++){
        u64 fx = fold_mod(X[i], W, M);
        u64 ft = fold_mod(T[i], W, M);
        Sin = (Sin + (u128)fx*ft % M) % M;
    }
    for(u64 y: Y){ Sout = (Sout + fold_mod(y, W, M)) % M; }
    return Sin==Sout;
}

// -------- Fault injection --------
std::vector<u64> inject_scf_matrix(const std::vector<u64>& C, const std::string& subtype,
                                   int bitwidth, u64 mod, int S, ThreadRNG& rng){
    auto R = C;
    int i = rng.randi(0,S-1), j = rng.randi(0,S-1);
    auto wrap = [&](u64 v){ return v % mod; };
    if(subtype=="SCF-BF"){
        int b = rng.randi(0, bitwidth-1);
        R[i*S+j] = wrap(R[i*S+j] ^ (1ull<<b));
    }else{ // SCF-MBU
        int K = (bitwidth>=4)? std::vector<int>{2,3,4}[rng.randi(0,2)] : 2;
        int start = std::max(0, rng.randi(0, std::max(0, bitwidth-K)));
        u64 mask = ((K>=64)?~0ull:((1ull<<K)-1ull)) << start;
        R[i*S+j] = wrap(R[i*S+j] ^ mask);
    }
    return R;
}
std::vector<u64> inject_scf_vector(const std::vector<u64>& X, const std::string& subtype,
                                   int bitwidth, u64 mod, ThreadRNG& rng){
    auto Y = X;
    int n = (int)Y.size();
    int idx = rng.randi(0, n-1);
    auto wrap = [&](u64 v){ return v % mod; };
    if(subtype=="SCF-BF"){
        int b = rng.randi(0, bitwidth-1);
        Y[idx] = wrap(Y[idx] ^ (1ull<<b));
    }else{ // SCF-MBU
        int K = (bitwidth<3)? 2 : std::vector<int>{2,3}[rng.randi(0,1)];
        int start = std::max(0, rng.randi(0, std::max(0, bitwidth-K)));
        u64 mask = ((K>=64)?~0ull:((1ull<<K)-1ull)) << start;
        Y[idx] = wrap(Y[idx] ^ mask);
    }
    return Y;
}
std::vector<u64> inject_mcf_matrix(const std::vector<u64>& A, const std::vector<u64>& B,
                                   const std::vector<u64>& C, const std::string& subtype,
                                   int bitwidth, u64 mod, int S, ThreadRNG& rng){
    auto R = C;
    auto wrap = [&](u64 v){ return v % mod; };
    if(subtype=="MCF-PPE"){
        int i = rng.randi(0,S-1), j=rng.randi(0,S-1);
        int b = rng.randi(0, bitwidth-1);
        R[i*S+j] = wrap(R[i*S+j] ^ (1ull<<b));
    }else if(subtype=="MCF-CTE"){
        int i = rng.randi(0,S-1), j=rng.randi(0,S-1);
        int lo = std::max(1, bitwidth/2);
        int b = rng.randi(lo, std::max(lo, bitwidth-1));
        R[i*S+j] = wrap(R[i*S+j] ^ (1ull<<b));
    }else{ // MCF-CLE
        int i = rng.randi(0,S-1);
        int k0 = rng.randi(0,S-1);
        int sgn = rng.coin()? +1 : -1;
        int mfac = rng.coin()? 1 : 2;
        u64 aik = A[i*S + k0];
        for(int j=0;j<S;j++){
            u64 delta = (u64)(((__int128)sgn * mfac * (u128)aik * B[k0*S + j]) % mod + mod) % mod;
            R[i*S + j] = wrap(R[i*S + j] + delta);
        }
    }
    return R;
}
std::vector<u64> inject_mcf_vector(const std::vector<u64>& X, const std::vector<u64>& T,
                                   const std::vector<u64>& Y, const std::string& subtype,
                                   int bitwidth, u64 mod, ThreadRNG& rng){
    auto R = Y;
    int n = (int)R.size();
    int idx = rng.randi(0, n-1);
    auto wrap = [&](u64 v){ return v % mod; };
    if(subtype=="MCF-PPE"){
        int b = rng.randi(0, bitwidth-1);
        R[idx] = wrap(R[idx] ^ (1ull<<b));
    }else if(subtype=="MCF-CTE"){
        int lo = std::max(1, bitwidth-2);
        int b = rng.randi(lo, std::max(lo, bitwidth-1));
        R[idx] = wrap(R[idx] ^ (1ull<<b));
    }else{ // MCF-CLE
        int k = std::vector<int>{-1, 2, -2}[rng.randi(0,2)];
        u64 wrong = (u64)((u128)X[idx]*T[idx] % mod);
        long long v = (long long)((k%mod+mod)%mod);
        u64 val = (u64)(( (u128)((v% (long long)mod + (long long)mod)% (long long)mod) * wrong) % mod);
        R[idx] = wrap(val);
    }
    return R;
}

// -------- One trial --------
struct TrialResult { bool det1=false, det2=false, det3=false; int inj_stage=0; };

TrialResult run_one_trial(const std::string& ftype, int S, u64 P, int W, u64 M, ThreadRNG& rng){
    int bitwidth_p = 64 - __builtin_clzll(P);

    // Stage 1 random data
    std::vector<u64> A1(S*S), B1(S*S), C1(S*S);
    for(int i=0;i<S*S;i++){ A1[i]=rng.randint(0,P-1); B1[i]=rng.randint(0,P-1); }
    matmul_mod(A1,B1,C1,S,P);

    // Stage 2 data
    std::vector<u64> X(S*S), T(S*S), Y(S*S);
    for(int i=0;i<S*S;i++){ X[i] = C1[i] % M; }
    for(int i=0;i<S*S;i++){ T[i] = rng.randint(0, M-1); }
    for(int i=0;i<S*S;i++){ Y[i] = (u64)((u128)X[i]*T[i] % M); }

    // Stage 3 data
    std::vector<u64> B2 = Y; for(u64& v: B2){ v %= P; }
    std::vector<u64> A2(S*S), C3(S*S);
    for(int i=0;i<S*S;i++){ A2[i]=rng.randint(0,P-1); }
    matmul_mod(A2,B2,C3,S,P);

    int inject_stage = rng.randi(1,3);
    bool det1=false, det2=false, det3=false;

    // Stage 1 protection
    if(inject_stage==1){
        std::vector<u64> C1f;
        if(ftype.rfind("SCF",0)==0) C1f = inject_scf_matrix(C1, ftype, bitwidth_p, P, S, rng);
        else                        C1f = inject_mcf_matrix(A1,B1,C1, ftype, bitwidth_p, P, S, rng);
        det1 = !matmul_with_protection(A1,B1,C1f,S,P);
    }else{
        (void)matmul_with_protection(A1,B1,C1,S,P);
    }

    // Stage 2 protection
    if(inject_stage==2){
        std::vector<u64> Yf;
        if(ftype.rfind("SCF",0)==0) Yf = inject_scf_vector(Y, ftype, W, M, rng);
        else                        Yf = inject_mcf_vector(X,T,Y, ftype, W, M, rng);
        det2 = !elementwise_with_fold(X,T,Yf,W,M);
    }else{
        (void)elementwise_with_fold(X,T,Y,W,M);
    }

    // Stage 3 protection
    if(inject_stage==3){
        std::vector<u64> C3f;
        if(ftype.rfind("SCF",0)==0) C3f = inject_scf_matrix(C3, ftype, bitwidth_p, P, S, rng);
        else                        C3f = inject_mcf_matrix(A2,B2,C3, ftype, bitwidth_p, P, S, rng);
        det3 = !matmul_with_protection(A2,B2,C3f,S,P);
    }else{
        (void)matmul_with_protection(A2,B2,C3,S,P);
    }

    return {det1,det2,det3,inject_stage};
}

// -------- Monte Carlo (OpenMP) --------
struct MCOut { uint64_t inj[3]{0,0,0}; uint64_t und[3]{0,0,0}; };

MCOut monte_carlo(const std::string& ftype, long long trials, int S, u64 P, int W, u64 M, uint64_t seed){
    MCOut out;
    #pragma omp parallel
    {
        MCOut local;
        uint64_t tid =
        #ifdef _OPENMP
            (uint64_t)omp_get_thread_num();
        #else
            0;
        #endif
        ThreadRNG rng(seed ^ (0x9E3779B97F4A7C15ull * (tid+1)));

        #pragma omp for schedule(static)
        for(long long t=0; t<trials; ++t){
            auto r = run_one_trial(ftype,S,P,W,M,rng);
            int idx = r.inj_stage-1;
            local.inj[idx] += 1;
            if(idx==0 && !r.det1) local.und[0] += 1;
            if(idx==1 && !r.det2) local.und[1] += 1;
            if(idx==2 && !r.det3) local.und[2] += 1;
        }

        #pragma omp critical
        {
            for(int i=0;i<3;i++){ out.inj[i] += local.inj[i]; out.und[i] += local.und[i]; }
        }
    }
    return out;
}

// -------- Main --------
int main(int argc, char** argv){
    Args a = parse_args(argc, argv);

    int S = (int)std::sqrt((double)a.N);
    if(S*S != a.N){ std::fprintf(stderr,"N must be a perfect square (N=S*S)\n"); return 1; }
    if(a.W < 2 || a.W > 30){ std::fprintf(stderr,"--W must be in [2,30]\n"); return 1; }

    ThreadRNG rng(a.seed);
    u64 P = a.p_override? a.p_override : find_prime_with_bitlen(a.pbits, rng);
    if(!is_probable_prime(P)){ std::fprintf(stderr,"Provided/Found P is not prime.\n"); return 1; }
    if(a.p_override && (int)(64-__builtin_clzll(P)) != a.pbits){
        std::fprintf(stderr,"[warn] --p bit-length %d != --pbits %d\n",
            (int)(64-__builtin_clzll(P)), a.pbits);
    }
    u64 M = (a.W==64)? ~0ull : ((1ull<<a.W)-1ull);

    std::printf("[config] Pbits=%d, P=%llu\n", (int)(64-__builtin_clzll(P)), (unsigned long long)P);
    std::printf("[config] W=%d, M=2^W-1=%llu\n", a.W, (unsigned long long)M);
    std::printf("[config] N=%d => S=%d, trials=%lld, seed=%llu\n", a.N, S, a.trials, (unsigned long long)a.seed);

    const std::vector<std::pair<std::string,std::string>> fault_types = {
        {"SCF-BF","storage_single_bit"},
        {"SCF-MBU","storage_multi_bit"},
        {"MCF-PPE","mul_partial_product"},
        {"MCF-CTE","mul_carry_tree"},
        {"MCF-CLE","mul_control_logic"},
    };

    for(const auto& ft: fault_types){
        auto out = monte_carlo(ft.first, a.trials, S, P, a.W, M, a.seed);
        std::printf("[%s - %s]\n", ft.first.c_str(), ft.second.c_str());
        for(int s=0;s<3;s++){
            double prob = out.inj[s] ? (double)out.und[s]/(double)out.inj[s] : 0.0;
            std::printf("  stage%d: injected=%llu, undetected=%llu, collision_prob=%.6f\n",
                s+1,
                (unsigned long long)out.inj[s],
                (unsigned long long)out.und[s],
                prob);
        }
    }
    return 0;
}

