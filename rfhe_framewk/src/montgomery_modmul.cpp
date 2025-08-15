// g++ -O3 -std=c++17 -fopenmp montgomery_mc.cpp -o montgomery_mc
// 需要 Boost 头文件（header-only） multiprecision

#include <bits/stdc++.h>
#include <omp.h>
#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using boost::multiprecision::cpp_int;

// ==================== 超参数 ====================
static constexpr uint32_t BITWIDTH_PRIME = 37;
static constexpr size_t   VECTOR_LEN     = 8192;
static constexpr int      TRIALS         = 10000;

static const vector<int> FOLD_WIDTHS = []{
    vector<int> v; for(int w=2; w<=32; w+=2) v.push_back(w); return v;
}();

// (USE_M_CHECK, USE_MP_CHECK, USE_FINAL_CHECK)
struct Scheme { bool use_m, use_mp, use_final; };
static const vector<Scheme> SCHEMES = {
    {true,  false, false},
    {false, true,  false},
    {false, false, true},
    {true,  false, true},
};

// 2D 注入
static constexpr int INJECT_ELEM_COUNT = 1;   // 每次注入多少个元素
static constexpr int BITFLIPS_PER_ELEM = 1;   // 每个元素翻转多少位
// =================================================

// -------- 64 位 Miller-Rabin --------
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
static uint64_t gen_prime_bitwidth(std::mt19937_64 &rng, uint32_t bitwidth){
    while(true){
        uint64_t p = 0;
        if(bitwidth >= 64){
            p = (rng() | (1ULL<<63)) | 1ULL;
        }else{
            uint64_t top  = 1ULL << (bitwidth-1);
            uint64_t mask = (bitwidth==64)? ~0ULL : ((1ULL<<bitwidth)-1);
            p = (rng() & mask) | top | 1ULL;
        }
        if(is_prime_u64(p)) return p;
    }
}

// -------- 工具 --------
static inline size_t bitlen(const cpp_int &x){
    if(x==0) return 0;
    return boost::multiprecision::msb(x) + 1;
}
static inline uint32_t bitlen_u64(uint64_t x){
    return x ? (64u - (uint32_t)__builtin_clzll(x)) : 0u;
}

static cpp_int fold_mod_sum(const cpp_int &sum, int width){
    // 对和做按 (2^w-1) 折叠
    cpp_int x = sum;
    cpp_int M = (cpp_int(1) << width) - 1;
    cpp_int s = 0;
    while(x > 0){
        s += (x & M);
        x >>= width;
    }
    s %= M;
    return s;
}
static inline cpp_int flip_bits(const cpp_int &v, const vector<size_t> &positions){
    cpp_int out = v;
    for(size_t pos: positions) out ^= (cpp_int(1) << pos);
    return out;
}

// 注入记录
struct InjectInfo { size_t idx; cpp_int before; cpp_int after; vector<size_t> bits; };

// 多元素多位注入，返回新向量与每个注入元素的 (idx,before,after)
static pair<vector<cpp_int>, vector<InjectInfo>>
inject_multi(const vector<cpp_int> &values, int elem_count, int bits_per_elem, std::mt19937_64 &rng)
{
    if(values.empty() || elem_count<=0) return {values, {}};

    int n = (int)values.size();
    elem_count = min(elem_count, n);

    vector<int> idxs(n); iota(idxs.begin(), idxs.end(), 0);
    shuffle(idxs.begin(), idxs.end(), rng);
    idxs.resize(elem_count);

    vector<cpp_int> out = values;
    vector<InjectInfo> infos; infos.reserve(elem_count);

    for(int idx : idxs){
        cpp_int v = out[idx];
        size_t bl = max<size_t>(1, bitlen(v));
        int k = min(bits_per_elem, (int)bl);

        unordered_set<size_t> used;
        vector<size_t> pos; pos.reserve(k);
        while((int)pos.size() < k){
            size_t p = (size_t)(rng() % bl);
            if(used.insert(p).second) pos.push_back(p);
        }
        cpp_int before = v;
        cpp_int after  = flip_bits(v, pos);
        out[idx] = after;
        infos.push_back({(size_t)idx, before, after, std::move(pos)});
    }
    return {std::move(out), std::move(infos)};
}

// -------- Montgomery 主体 --------
struct MontConsts {
    uint64_t p;      // modulus
    uint32_t k;      // bitlen(p)
    uint64_t R;      // 1<<k
    uint64_t m_prime;// (-p^{-1}) mod R
};

// 计算 (-p^{-1}) mod 2^k ；用牛顿提升
static uint64_t inv_mod_2k(uint64_t p, uint32_t k){
    uint64_t inv = 1; // mod 2
    for(uint32_t bits=1; bits<k; ++bits){
        uint64_t mod = (bits>=63)? 0ULL : (1ULL << (bits+1)); // 2^(bits+1)
        if(mod==0) { inv = inv; break; }
        uint64_t tmp = (2 - ( (__uint128_t)p * inv ) % mod) % mod;
        inv = ( (__uint128_t)inv * tmp ) % mod;
    }
    if(k < 64) inv &= ((1ULL<<k) - 1);
    return inv;
}
static MontConsts compute_montgomery_constants(uint64_t p){
    uint32_t k = bitlen_u64(p);
    uint64_t R = 1ULL << k;
    uint64_t inv = inv_mod_2k(p, k);                // p*inv ≡ 1 (mod 2^k)
    uint64_t m_prime = ((~inv) + 1ULL) & (R - 1);   // (-inv) mod 2^k
    return {p, k, R, m_prime};
}

static vector<cpp_int> montgomery_reduce_list(
    const vector<cpp_int> &T_list, const MontConsts &mc)
{
    const uint64_t maskR = mc.R - 1;
    vector<cpp_int> out; out.reserve(T_list.size());

    vector<uint64_t> lowT(T_list.size());
    for(size_t i=0;i<T_list.size();++i)
        lowT[i] = (uint64_t)(T_list[i] & maskR).convert_to<uint64_t>();

    vector<cpp_int> prod_m(T_list.size());
    for(size_t i=0;i<T_list.size();++i)
        prod_m[i] = cpp_int(lowT[i]) * mc.m_prime;

    vector<uint64_t> m(T_list.size());
    for(size_t i=0;i<T_list.size();++i)
        m[i] = (uint64_t)(prod_m[i] & maskR).convert_to<uint64_t>();

    vector<cpp_int> prod_mp(T_list.size());
    for(size_t i=0;i<T_list.size();++i)
        prod_mp[i] = cpp_int(m[i]) * mc.p;

    for(size_t i=0;i<T_list.size();++i){
        cpp_int Tval = T_list[i] + prod_mp[i];
        cpp_int u2 = Tval >> mc.k;
        cpp_int u  = (u2 >= mc.p) ? (u2 - mc.p) : u2;
        out.push_back(u);
    }
    return out;
}

// 故障 + ECC（差分更新 ECC）
static tuple<bool,bool,vector<bool>, vector<cpp_int>>
montgomery_reduce_list_ecc_inject_detect(
    const vector<cpp_int> &T_list_in, const MontConsts &mc, int width,
    int elem_count, int bits_per_elem, // 2D
    bool use_m_check, bool use_mp_check, bool use_final_check,
    std::mt19937_64 &rng)
{
    const uint64_t maskR = mc.R - 1;

    // 预计算 lowT, prod_m 及其折叠和
    vector<uint64_t> lowT(T_list_in.size());
    for(size_t i=0;i<T_list_in.size();++i)
        lowT[i] = (uint64_t)(T_list_in[i] & maskR).convert_to<uint64_t>();

    vector<cpp_int> prod_m(T_list_in.size());
    cpp_int sum_prod_m = 0;
    for(size_t i=0;i<T_list_in.size();++i){
        prod_m[i] = cpp_int(lowT[i]) * mc.m_prime;
        sum_prod_m += prod_m[i];
    }
    cpp_int ecc_m_before = use_m_check ? fold_mod_sum(sum_prod_m, width) : 0;

    // m, prod_mp 及其折叠和
    vector<uint64_t> m(T_list_in.size());
    for(size_t i=0;i<T_list_in.size();++i)
        m[i] = (uint64_t)(prod_m[i] & maskR).convert_to<uint64_t>();

    vector<cpp_int> prod_mp(T_list_in.size());
    cpp_int sum_prod_mp = 0;
    for(size_t i=0;i<T_list_in.size();++i){
        prod_mp[i] = cpp_int(m[i]) * mc.p;
        sum_prod_mp += prod_mp[i];
    }
    cpp_int ecc_mp_before = use_mp_check ? fold_mod_sum(sum_prod_mp, width) : 0;

    // 50/50 选注入阶段
    bool inject_m = ((rng() & 1ULL) != 0);

    if(inject_m){
        auto inj = inject_multi(prod_m, elem_count, bits_per_elem, rng);
        // 差分更新 sum_prod_m
        for(const auto &info : inj.second){
            sum_prod_m += info.after;
            sum_prod_m -= info.before;
        }
        prod_m = std::move(inj.first);

        // 重新 m、prod_mp 和 sum_prod_mp（仅受改动 idx 影响可再差分；这里简化整段重算一次 m 与 mp，仍比双折叠全量省一次）
        for(size_t i=0;i<T_list_in.size();++i)
            m[i] = (uint64_t)(prod_m[i] & maskR).convert_to<uint64_t>();
        sum_prod_mp = 0;
        for(size_t i=0;i<T_list_in.size();++i){
            prod_mp[i] = cpp_int(m[i]) * mc.p;
            sum_prod_mp += prod_mp[i];
        }
    }else{
        // 注入在 mp
        auto inj = inject_multi(prod_mp, elem_count, bits_per_elem, rng);
        for(const auto &info : inj.second){
            sum_prod_mp += info.after;
            sum_prod_mp -= info.before;
        }
        prod_mp = std::move(inj.first);
    }

    bool detect_m  = false;
    bool detect_mp = false;
    if(use_m_check && inject_m){
        cpp_int ecc_m_after = fold_mod_sum(sum_prod_m, width);
        detect_m = (ecc_m_after != ecc_m_before);
    }
    if(use_mp_check){
        cpp_int ecc_mp_after = fold_mod_sum(sum_prod_mp, width);
        detect_mp = (ecc_mp_after != ecc_mp_before);
    }

    vector<bool> detect_final_flags; detect_final_flags.reserve(T_list_in.size());
    vector<cpp_int> out_fault; out_fault.reserve(T_list_in.size());
    for(size_t i=0;i<T_list_in.size();++i){
        cpp_int Tval = T_list_in[i] + prod_mp[i];
        bool flag = false;
        if(use_final_check){
            flag = ((Tval & maskR) != 0); // 低 k 位非零
        }
        cpp_int u2 = Tval >> mc.k;
        cpp_int u  = (u2 >= mc.p) ? (u2 - mc.p) : u2;
        out_fault.push_back(u);
        detect_final_flags.push_back(flag);
    }

    return {detect_m, detect_mp, detect_final_flags, out_fault};
}

// -------- Monte Carlo（并行 + 进度条 + 容器复用） --------
struct Counts { uint64_t TP=0, FP=0, TN=0, FN=0; };

static Counts run_experiment(int fold_width, const Scheme &sch,
                             const MontConsts &mc, uint64_t p_u64)
{
    Counts tot;
    omp_set_num_threads(16);

    // 进度条
    const int step = max(1, TRIALS / 20); // 每 5% 一次
    std::atomic<int> done{0};

    // 基础种子
    uint64_t base_seed = std::mt19937_64(std::random_device{}())();

    #pragma omp parallel
    {
        uint64_t s = base_seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(omp_get_thread_num()+1));
        std::mt19937_64 rng(s);

        uint64_t TP=0, FP=0, TN=0, FN=0;

        // 复用容器
        vector<cpp_int> T_list;      T_list.reserve(VECTOR_LEN);
        vector<cpp_int> out_correct; out_correct.reserve(VECTOR_LEN);

        #pragma omp for schedule(static)
        for(int trial=0; trial<TRIALS; ++trial){
            T_list.clear(); out_correct.clear();

            // 随机构造 T_list = a*b
            for(size_t i=0;i<VECTOR_LEN;++i){
                uint64_t a = 1 + (rng() % (p_u64 - 1));
                uint64_t b = 1 + (rng() % (p_u64 - 1));
                T_list.push_back( cpp_int(a) * cpp_int(b) );
            }

            // 正确输出
            out_correct = montgomery_reduce_list(T_list, mc);

            // 故障 + 检测
            auto res = montgomery_reduce_list_ecc_inject_detect(
                T_list, mc, fold_width,
                INJECT_ELEM_COUNT, BITFLIPS_PER_ELEM,
                sch.use_m, sch.use_mp, sch.use_final,
                rng
            );
            bool detect_m   = get<0>(res);
            bool detect_mp  = get<1>(res);
            const vector<bool> &final_flags = get<2>(res);
            const vector<cpp_int> &out_fault = get<3>(res);

            bool detected_any = detect_m || detect_mp ||
               (sch.use_final && std::any_of(final_flags.begin(), final_flags.end(), [](bool f){return f;}));

            bool harmful = (out_fault != out_correct);
            if(harmful){
                if(detected_any) ++TP; else ++FN;
            }else{
                if(detected_any) ++FP; else ++TN;
            }

            // 进度
            int d = ++done;
            if ((d % step) == 0 && omp_get_thread_num()==0) {
                cerr << "\rProgress(fw=" << fold_width << "): " << (d * 100 / TRIALS) << "%" << flush;
            }
        }

        #pragma omp atomic
        tot.TP += TP;
        #pragma omp atomic
        tot.FP += FP;
        #pragma omp atomic
        tot.TN += TN;
        #pragma omp atomic
        tot.FN += FN;
    }
    // cerr << "\rProgress(fw=" << fold_width << "): 100%\n";
    return tot;
}

int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 固定一次素数与常量
    std::mt19937_64 seeder(std::random_device{}());
    uint64_t p_u64 = gen_prime_bitwidth(seeder, BITWIDTH_PRIME);
    MontConsts mc = compute_montgomery_constants(p_u64);

    cout << "FOLD_WIDTH, USE_M, USE_MP, USE_FINAL, DetectionRate, TP, FP, TN, FN\n";
    for(int fw: FOLD_WIDTHS){
        for(const auto &sch: SCHEMES){
            Counts c = run_experiment(fw, sch, mc, p_u64);
            uint64_t harmful = c.TP + c.FN;
            double rate = harmful ? (double)c.TP / (double)harmful : 1.0;
            cout << fw << ", "
                 << (int)sch.use_m << ", "
                 << (int)sch.use_mp << ", "
                 << (int)sch.use_final << ", "
                 << fixed << setprecision(4) << rate << ", "
                 << c.TP << ", " << c.FP << ", " << c.TN << ", " << c.FN << "\n";
        }
    }
    return 0;
}
