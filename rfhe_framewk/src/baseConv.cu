// file: crt_cuda.cu
#include <cuda_runtime.h>
#include <vector>
#include <unordered_set>
#include <random>
#include <iostream>
#include <cassert>
#include <cstdint>
#include <stdexcept>

// ===================== Miller–Rabin 素性测试 & 下一个素数 =====================
static const uint64_t MR_BASES[] = {
    2ULL, 325ULL, 9375ULL, 28178ULL,
    450775ULL, 9780504ULL, 1795265022ULL
};

inline uint64_t mod_mul(uint64_t a, uint64_t b, uint64_t mod) {
    __uint128_t z = ( __uint128_t ) a * b;
    return (uint64_t)(z % mod);
}

uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    while (exp) {
        if (exp & 1) res = mod_mul(res, base, mod);
        base = mod_mul(base, base, mod);
        exp >>= 1;
    }
    return res;
}

bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    for (uint64_t p : {2ULL,3ULL,5ULL,7ULL,11ULL,13ULL,17ULL,19ULL,23ULL}) {
        if (n % p == 0) return n == p;
    }
    uint64_t d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint64_t a : MR_BASES) {
        if (a % n == 0) break;
        uint64_t x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (uint64_t r = 1; r < s; ++r) {
            x = mod_mul(x, x, n);
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

uint64_t next_prime_u64(uint64_t cand) {
    if (cand <= 2) return 2;
    if ((cand & 1) == 0) ++cand;
    while (!is_prime_u64(cand)) cand += 2;
    return cand;
}
// =============================================================

// 扩展欧几里得，返回 g=gcd(a,b)，并求 x,y 使 a*x + b*y = g
int64_t ext_gcd(int64_t a, int64_t b, int64_t &x, int64_t &y) {
    if (b == 0) {
        x = 1; y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t g = ext_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// 计算 a 对 m 的模逆，要求 gcd(a,m)=1
uint64_t mod_inv(uint64_t a, uint64_t m) {
    int64_t x, y;
    int64_t g = ext_gcd((int64_t)(a % m), (int64_t)m, x, y);
    if (g != 1) throw std::runtime_error("modular inverse does not exist");
    x %= (int64_t)m;
    if (x < 0) x += m;
    return (uint64_t)x;
}

// ------------------ CUDA 核函数 ------------------
__global__ void crt_kernel(
    const uint64_t* __restrict__ residues, // m × N, row-major
    const uint64_t* __restrict__ moduli,   // m
    const unsigned __int128* __restrict__ p_prefix, // m: p0, p0*p1, ...
    const uint64_t* __restrict__ inv_prefix,        // m
    uint64_t* __restrict__ x_lo,      // N 输出：低 64 位
    uint64_t* __restrict__ x_hi,      // N 输出：高余下位
    int m,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Garner 系数 c[0..m-1]
    unsigned long long c[8];  // 假设 m ≤ 8
    c[0] = residues[0 * N + i];

    for (int j = 1; j < m; j++) {
        unsigned long long t = residues[j * N + i] % moduli[j];
        for (int k = 0; k < j; k++) {
            unsigned __int128 prod = (unsigned __int128)c[k] * p_prefix[k];
            unsigned long long r = (unsigned long long)(prod % moduli[j]);
            t = (t + moduli[j] - r) % moduli[j];
        }
        t = (unsigned long long)(((__int128)t * inv_prefix[j]) % moduli[j]);
        c[j] = t;
    }

    unsigned __int128 x128 = 0;
    for (int k = 0; k < m; k++) {
        x128 += (unsigned __int128)c[k] * p_prefix[k];
    }

    x_lo[i] = (uint64_t)(x128 & 0xFFFFFFFFFFFFFFFFULL);
    x_hi[i] = (uint64_t)(x128 >> 64);
}

// ---------------------- Host 代码 ----------------------
int main() {
    // const int limbs    = 4;
    // const int bitwidth = 40;
    // const int N        = 32768;

    const int limbs    = 3;
    const int bitwidth = 10;
    const int N        = 1;

    // 1) 生成 moduli
    std::mt19937_64 rng(12345);
    std::vector<uint64_t> h_moduli;
    std::unordered_set<uint64_t> seen;
    while ((int)h_moduli.size() < limbs) {
        uint64_t cand = (rng() & ((1ULL<<bitwidth)-1))
                      | (1ULL<<(bitwidth-1))
                      | 1ULL;
        uint64_t p = next_prime_u64(cand);
        if (p >> bitwidth) continue;
        if (seen.insert(p).second) h_moduli.push_back(p);
    }

    // 2) 生成 residues[m][N]
    std::vector<uint64_t> h_residues(limbs * N);
    for (int j = 0; j < limbs; j++) {
        uint64_t p = h_moduli[j];
        for (int i = 0; i < N; i++) {
            h_residues[j*N + i] = rng() % p;
        }
        uint64_t sum = 0;
        for (int i = 0; i < N; i++) sum = (sum + h_residues[j*N + i]) % p;
        h_residues[j*N + (N-1)] = sum;
    }

    // 3) 预计算 Garner 前缀积 & 逆元
    __int128 P = 1;
    for (auto p : h_moduli) P *= p;
    std::vector<unsigned __int128> h_p_prefix(limbs);
    std::vector<uint64_t>          h_inv_prefix(limbs);
    h_p_prefix[0] = 1;
    for (int j = 1; j < limbs; j++) {
        h_p_prefix[j] = h_p_prefix[j-1] * h_moduli[j-1];
    }
    for (int j = 1; j < limbs; j++) {
        uint64_t a_mod = (uint64_t)(h_p_prefix[j] % h_moduli[j]);
        h_inv_prefix[j] = mod_inv(a_mod, h_moduli[j]);
    }

    // 4) 分配 & 拷贝到 Device
    uint64_t *d_res, *d_mod, *d_inv, *d_xlo, *d_xhi;
    unsigned __int128* d_pref;
    cudaMalloc(&d_res,  sizeof(uint64_t)*limbs*N);
    cudaMalloc(&d_mod,  sizeof(uint64_t)*limbs);
    cudaMalloc(&d_pref, sizeof(unsigned __int128)*limbs);
    cudaMalloc(&d_inv,  sizeof(uint64_t)*limbs);
    cudaMalloc(&d_xlo,  sizeof(uint64_t)*N);
    cudaMalloc(&d_xhi,  sizeof(uint64_t)*N);

    cudaMemcpy(d_res,  h_residues.data(), sizeof(uint64_t)*limbs*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod,  h_moduli.data(),    sizeof(uint64_t)*limbs,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_pref, h_p_prefix.data(),  sizeof(unsigned __int128)*limbs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inv,  h_inv_prefix.data(), sizeof(uint64_t)*limbs,    cudaMemcpyHostToDevice);

    // 5) Launch kernel
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    crt_kernel<<<blocks, threads>>>(
        d_res, d_mod, d_pref, d_inv,
        d_xlo, d_xhi,
        limbs, N
    );
    cudaDeviceSynchronize();

    // 6) 拷回 & 验证
    std::vector<uint64_t> h_xlo(N), h_xhi(N);
    cudaMemcpy(h_xlo.data(), d_xlo, sizeof(uint64_t)*N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_xhi.data(), d_xhi, sizeof(uint64_t)*N, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        unsigned __int128 xi = ((unsigned __int128)h_xhi[i] << 64) | h_xlo[i];
        for (int j = 0; j < limbs; j++) {
            assert((uint64_t)(xi % h_moduli[j]) == h_residues[j*N + i]);
        }
    }
    std::cout << "CRT reconstruction validated on GPU!" << std::endl;
    std::vector<__int128> recon(N);
    for (int i = 0; i < N; i++) {
        recon[i] = ( (__int128)h_xhi[i] << 64 ) | h_xlo[i];
    }

    // ECC 验证：最后一个元素应当等于前面所有元素之和 mod P
    __int128 sum = 0;
    for (int i = 0; i < N-1; i++) {
        sum = (sum + recon[i]) % P;
    }
    assert( recon[N-1] == sum );
    std::cout << "ECC check passed!\n";

    // 7) 释放资源（可选）
    cudaFree(d_res);
    cudaFree(d_mod);
    cudaFree(d_pref);
    cudaFree(d_inv);
    cudaFree(d_xlo);
    cudaFree(d_xhi);

    return 0;
}
