// baseConv.cpp
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <set>
#include <cstdint>
#include <climits>

// 128-bit 整数别名
using uint128 = unsigned __int128;
using uint64 = uint64_t;

// --- Miller-Rabin 素性测试（确定性，适用于 64 位） ---
uint64 mod_pow(uint64 a, uint64 d, uint64 m) {
    uint128 res = 1, base = a % m;
    while (d) {
        if (d & 1) res = (res * base) % m;
        base = (uint128)base * base % m;
        d >>= 1;
    }
    return (uint64)res;
}

bool is_prime(uint64 n) {
    if (n < 2) return false;
    for (uint64 p : {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL}) {
        if (n % p == 0) return n == p;
    }
    uint64 d = n - 1, s = 0;
    while ((d & 1) == 0) { d >>= 1; ++s; }
    for (uint64 a : {2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL}) {
        if (a % n == 0) continue;
        uint64 x = mod_pow(a, d, n);
        if (x == 1 || x == n - 1) continue;
        bool comp = true;
        for (uint64 r = 1; r < s; ++r) {
            x = (uint128)x * x % n;
            if (x == n - 1) { comp = false; break; }
        }
        if (comp) return false;
    }
    return true;
}

uint64 next_prime(uint64 cand) {
    if ((cand & 1) == 0) ++cand;
    while (!is_prime(cand)) cand += 2;
    return cand;
}

// 扩展欧几里得，计算 a * x + m * y = gcd(a, m)
int64_t egcd(int64_t a, int64_t b, int64_t &x, int64_t &y) {
    if (b == 0) { x = 1; y = 0; return a; }
    int64_t x1, y1;
    int64_t g = egcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return g;
}

// 计算 a 在模 m 下的乘法逆元
uint64 mod_inv(uint64 a, uint64 m) {
    int64_t x, y;
    int64_t g = egcd(a, m, x, y);
    if (g != 1) throw std::runtime_error("No inverse");
    x %= (int64_t)m;
    if (x < 0) x += m;
    return (uint64)x;
}

// 生成 limbs 个 bitwidth 位素数
std::vector<uint64> generate_crt_primes(int limbs, int bitwidth) {
    if (bitwidth < 10) throw std::invalid_argument("bitwidth must be at least 10");
    std::mt19937_64 gen(std::random_device{}());
    std::set<uint64> seen;
    std::vector<uint64> primes;
    uint64 mask = (bitwidth == 64 ? ~0ULL : ((1ULL << bitwidth) - 1));
    while ((int)primes.size() < limbs) {
        uint64 cand = (gen() & mask) | (1ULL << (bitwidth - 1)) | 1ULL;
        uint64 p = next_prime(cand);
        int bl = p ? 64 - __builtin_clzll(p) : 1;
        if (bl == bitwidth && seen.insert(p).second) {
            primes.push_back(p);
        }
    }
    return primes;
}

// 生成 poly_dim + 1 长度的残余向量（最后一位为 ECC 校验和）
std::vector<std::vector<uint64>> generate_residues(
    const std::vector<uint64> &moduli, int poly_dim)
{
    std::mt19937_64 gen(std::random_device{}());
    std::vector<std::vector<uint64>> residues;
    for (auto p : moduli) {
        std::uniform_int_distribution<uint64> dist(0, p - 1);
        std::vector<uint64> arr(poly_dim + 1);
        uint64 sum = 0;
        for (int i = 0; i < poly_dim; ++i) {
            arr[i] = dist(gen);
            sum = (sum + arr[i]) % p;
        }
        arr[poly_dim] = sum;
        residues.push_back(arr);
    }
    return residues;
}

// 随机在矩阵中选 m 个元素，然后总共翻转 n 位
void flip_n_bits_across_m_elements_2d_inplace(
    std::vector<std::vector<uint64>> &matrix,
    int m, int n)
{
    int rows = matrix.size();
    int cols = matrix[0].size();
    int total = rows * cols;
    if (m < 1 || m > total) throw std::invalid_argument("Invalid m");
    std::vector<std::pair<int,int>> pos;
    pos.reserve(total);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            pos.emplace_back(i, j);
    std::mt19937_64 gen(std::random_device{}());
    std::shuffle(pos.begin(), pos.end(), gen);
    pos.resize(m);

    std::vector<std::tuple<int,int,int>> bits;
    for (auto [r,c] : pos) {
        uint64 v = matrix[r][c];
        int w = v ? 64 - __builtin_clzll(v) : 1;
        for (int b = 0; b < w; ++b)
            bits.emplace_back(r, c, b);
    }
    if (n < 0 || n > (int)bits.size()) throw std::invalid_argument("Invalid n");
    std::shuffle(bits.begin(), bits.end(), gen);
    for (int i = 0; i < n; ++i) {
        auto [r,c,b] = bits[i];
        matrix[r][c] ^= (1ULL << b);
    }
}

// CRT 重建（返回长度为 N 的向量，使用 128 位运算避免溢出）
std::vector<uint128> crt_reconstruct(
    const std::vector<std::vector<uint64>> &residues,
    const std::vector<uint64> &moduli)
{
    int m = moduli.size();
    int N = residues[0].size();
    uint128 P = 1;
    for (auto p : moduli) P *= p;

    std::vector<uint128> hat_p(m);
    std::vector<uint64> inv_hat_p(m);
    for (int j = 0; j < m; ++j) {
        hat_p[j] = P / moduli[j];
        inv_hat_p[j] = mod_inv((uint64)(hat_p[j] % moduli[j]), moduli[j]);
    }

    std::vector<uint128> result(N);
    for (int i = 0; i < N; ++i) {
        uint128 x = 0;
        for (int j = 0; j < m; ++j) {
            uint128 term = (uint128)residues[j][i] * hat_p[j] % P;
            term = term * inv_hat_p[j] % P;
            x = (x + term) % P;
        }
        result[i] = x;
    }
    return result;
}

int main() {
    const int limbs         = 4;
    const int crt_bitwidth  = 20;
    const int poly_dim      = 1024;
    const int flip_bits     = 2;
    const int flip_elements = 2;
    const int epochs        = 100000;
    const int bar_width     = 50;

    uint64 ecc_pass = 0, ecc_fail = 0, crt_fail = 0;

    for (int e = 0; e < epochs; ++e) {
        // 进度条
        if (e % (epochs / bar_width == 0 ? 1 : epochs / bar_width) == 0) {
            int filled = e * bar_width / epochs;
            std::cout << "\r[";
            for (int i = 0; i < bar_width; ++i) std::cout << (i < filled ? '=' : ' ');
            std::cout << "] " << (e * 100 / epochs) << "%" << std::flush;
        }

        auto moduli   = generate_crt_primes(limbs, crt_bitwidth);
        auto residues = generate_residues(moduli, poly_dim);
        auto original = residues;
        flip_n_bits_across_m_elements_2d_inplace(residues, flip_elements, flip_bits);

        auto recon = crt_reconstruct(residues, moduli);

        bool crt_err = false;
        for (int i = 0; i < (int)recon.size(); ++i) {
            for (int j = 0; j < limbs; ++j) {
                if ((recon[i] % moduli[j]) != original[j][i]) {
                    crt_err = true; break;
                }
            }
            if (crt_err) break;
        }
        if (crt_err) ++crt_fail;

        uint128 P = 1; for (auto p : moduli) P *= p;
        uint128 sum = 0; for (int i = 0; i < poly_dim; ++i) sum = (sum + recon[i]) % P;
        if (sum == recon[poly_dim]) ++ecc_pass; else ++ecc_fail;
    }
    std::cout << "\r[" << std::string(bar_width, '=') << "] 100%\n";

    std::cout << "EPOCHS: "        << epochs    << "\n"
              << "ECC FAIL TIME: " << ecc_fail  << "\n"
              << "CRT FAIL TIME: " << crt_fail  << "\n";
    return 0;
}
