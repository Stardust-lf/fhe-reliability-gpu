#include "phantom.h"
#include <iostream>
#include <random>
#include <memory>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <cstdlib>

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

/*-------- 安全的加法取模（避免溢出） --------*/
__host__ __device__ inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t mod)
{
    uint64_t s = a + b;
    return (s < a || s >= mod) ? s - mod : s;
}

void test_nwt(size_t log_dim, size_t batch_size, int num_flips) {
    // 1. 计算 dim 和总可翻转比特数
    size_t dim = 1ULL << log_dim;  // N = 2^log_dim
    uint64_t total_elements = batch_size * dim;
    uint64_t total_bits = total_elements * 64ULL;  // 每个 uint64_t 有 64 个比特

    // 2. 检查 num_flips 是否合理
    if (static_cast<uint64_t>(num_flips) > total_bits) {
        std::cerr << "ERROR: num_flips (" << num_flips
                  << ") exceeds total possible bits (" << total_bits << ").\n";
        std::exit(1);
    }

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    // 3. Host 端生成模数列表
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));

    // 4. 复制到 Device 端的 DModulus 数组
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; ++i) {
        modulus.get()[i].set(
            h_modulus[i].value(),
            h_modulus[i].const_ratio()[0],
            h_modulus[i].const_ratio()[1]
        );
    }

    // 5. 构建并初始化 GPU 端的 DNTTTable（只需要正向 Twiddle）
    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size, s);
    for (size_t i = 0; i < batch_size; ++i) {
        NTT h_ntt_table(log_dim, h_modulus[i]);
        d_ntt_tables.set(
            &modulus.get()[i],
            h_ntt_table.get_from_root_powers().data(),
            h_ntt_table.get_from_root_powers_shoup().data(),
            /*逆向 Twiddle*/ nullptr, nullptr,
            /*n_inv*/ 0, 0,
            i, s
        );
    }

    // 6. Host 端初始化随机输入数组 h_data
    auto h_data = std::make_unique<uint64_t[]>(total_elements);
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (size_t i = 0; i < batch_size; ++i) {
        uint64_t mod = h_modulus[i].value();
        std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
        for (size_t j = 0; j < dim; ++j) {
            h_data.get()[i * dim + j] = dist(rng);
        }
    }

    // 7. 准备存储 clean 和 faulty 的 NTT 输出
    auto h_clean_ntt  = std::make_unique<uint64_t[]>(total_elements);
    auto h_faulty_ntt = std::make_unique<uint64_t[]>(total_elements);

    // 8. 将干净输入拷贝到 Device 并执行 2D NTT（只是前向部分）
    auto d_data = phantom::util::make_cuda_auto_ptr<uint64_t>(total_elements, s);
    cudaMemcpyAsync(
        d_data.get(),
        h_data.get(),
        total_elements * sizeof(uint64_t),
        cudaMemcpyHostToDevice, s
    );
    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, /*start_mod=*/0, s);
    cudaMemcpyAsync(
        h_clean_ntt.get(),
        d_data.get(),
        total_elements * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, s
    );
    cudaStreamSynchronize(s);

    // 9. 在 Host 端随机选择且“不重复”的 num_flips 个全局比特索引
    std::uniform_int_distribution<uint64_t> bit_index_dist(0, total_bits - 1);
    std::unordered_set<uint64_t> chosen_bits;
    chosen_bits.reserve(static_cast<size_t>(num_flips));

    while (static_cast<int>(chosen_bits.size()) < num_flips) {
        uint64_t candidate = bit_index_dist(rng);
        chosen_bits.insert(candidate);
    }

    // // 10. 对每个选中的全局比特索引进行翻转
    // std::cout << "[2D] Flipping " << num_flips << " unique random bit"
    //           << (num_flips == 1 ? "" : "s") << " in the data buffer...\n";

    // for (uint64_t global_bit_idx : chosen_bits) {
    //     uint64_t elem_idx = global_bit_idx / 64ULL;         // 第几个 uint64_t
    //     int bitpos       = static_cast<int>(global_bit_idx % 64ULL);  // uint64_t 内的哪一位
    //     uint64_t mask    = (1ULL << bitpos);
    //     h_data.get()[elem_idx] ^= mask;
    // }

    // 11. 翻转后拷贝到 Device 并再执行 2D NTT
    cudaMemcpyAsync(
        d_data.get(),
        h_data.get(),
        total_elements * sizeof(uint64_t),
        cudaMemcpyHostToDevice, s
    );
    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, /*start_mod=*/0, s);
    cudaMemcpyAsync(
        h_faulty_ntt.get(),
        d_data.get(),
        total_elements * sizeof(uint64_t),
        cudaMemcpyDeviceToHost, s
    );
    cudaStreamSynchronize(s);

    // 12. 比较 clean_ntt vs faulty_ntt，统计 bit-level 汉明距离和受影响的符号个数
    size_t total_hamming = 0;
    size_t symbol_mismatch_count = 0;

    for (size_t i = 0; i < total_elements; ++i) {
        uint64_t clean = h_clean_ntt.get()[i];
        uint64_t fault = h_faulty_ntt.get()[i];
        uint64_t diff  = clean ^ fault;
        if (diff != 0) {
            ++symbol_mismatch_count;
        }
        total_hamming += static_cast<size_t>(__builtin_popcountll(diff));
    }

    if (total_hamming != 0) {
        // 计算比特错误率和符号错误率
        double bit_error_rate = static_cast<double>(total_hamming) / static_cast<double>(total_bits);
        double symbol_error_rate = static_cast<double>(symbol_mismatch_count) / static_cast<double>(total_elements);

        std::cout << "ERROR! Total bitwise Hamming distance = "
                  << total_hamming
                  << " (bit error rate = " << bit_error_rate << ")\n";
        std::cout << "       Affected symbols = "
                  << symbol_mismatch_count
                  << "/" << total_elements
                  << " (symbol error rate = " << symbol_error_rate << ")\n";

        std::fprintf(
            stderr,
            "[FAULT DETECTED] Bit error: %zu/%zu = %.6f, Symbol error: %zu/%zu = %.6f\n",
            total_hamming, total_bits, bit_error_rate,
            symbol_mismatch_count, total_elements, symbol_error_rate
        );
    } else {
        std::cout << "ALL CORRECT\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <log_dim> <batch_size> <num_flips>\n";
        return 1;
    }

    size_t log_dim    = static_cast<size_t>(std::stoul(argv[1]));
    size_t batch_size = static_cast<size_t>(std::stoul(argv[2]));
    int num_flips     = std::stoi(argv[3]);

    if (num_flips < 1) {
        std::cerr << "Error: <num_flips> must be >= 1\n";
        return 1;
    }

    test_nwt(log_dim, batch_size, num_flips);
    return 0;
}
