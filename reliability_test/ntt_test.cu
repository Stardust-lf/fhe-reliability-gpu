#include "phantom.h"
#include <iostream>
#include <random>
#include <memory>
#include <unordered_set>
#include <vector>
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

void test_nwt(size_t log_dim, size_t batch_size, int num_flips, int num_target_symbols) {
    // 1. 计算 dim 和总可翻转比特数
    size_t dim = 1ULL << log_dim;  // N = 2^log_dim
    uint64_t total_elements = batch_size * dim;
    uint64_t total_bits = total_elements * 64ULL;  // 每个 uint64_t 有 64 个比特

    // 2. 检查参数合理性
    if (static_cast<uint64_t>(num_flips) > 64ULL) {
        std::cerr << "ERROR: num_flips (" << num_flips
                  << ") exceeds bits per symbol (64).\n";
        std::exit(1);
    }
    if (static_cast<uint64_t>(num_target_symbols) > total_elements) {
        std::cerr << "ERROR: num_target_symbols (" << num_target_symbols
                  << ") exceeds total symbols (" << total_elements << ").\n";
        std::exit(1);
    }

    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();

    // 3. Host 端生成模数列表
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));

    // 4. 复制到 Device 端的 DModulus 数组
    auto modulus = make_cuda_auto_ptr<DModulus>(batch_size, s);
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

    // 8. 将干净输入拷贝到 Device 并执行 2D NTT（前向部分）
    auto d_data = make_cuda_auto_ptr<uint64_t>(total_elements, s);
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

    // 9. 随机选择 num_target_symbols 个 symbol，并在每个 symbol 中翻转 num_flips 个 bit
    std::unordered_set<uint64_t> selected_symbols;
    std::uniform_int_distribution<uint64_t> symbol_dist(0, total_elements - 1);
    std::uniform_int_distribution<int> bitpos_dist(0, 63);

    while (static_cast<int>(selected_symbols.size()) < num_target_symbols) {
        selected_symbols.insert(symbol_dist(rng));
    }

    std::unordered_set<uint64_t> flipped_bits;
    for (uint64_t elem_idx : selected_symbols) {
        std::unordered_set<int> bit_positions;
        while (static_cast<int>(bit_positions.size()) < num_flips) {
            bit_positions.insert(bitpos_dist(rng));
        }
        for (int bitpos : bit_positions) {
            uint64_t global_bit_idx = elem_idx * 64ULL + bitpos;
            flipped_bits.insert(global_bit_idx);
        }
    }

    std::cout << "[2D] Flipping " << flipped_bits.size() 
              << " bits across " << selected_symbols.size()
              << " symbols...\n";

    // 10. 进行 bit flip
    for (uint64_t global_bit_idx : flipped_bits) {
        uint64_t elem_idx = global_bit_idx / 64ULL;
        int bitpos = static_cast<int>(global_bit_idx % 64ULL);
        uint64_t mask = (1ULL << bitpos);
        h_data.get()[elem_idx] ^= mask;
    }

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

    // 12. 比较 clean_ntt vs faulty_ntt，统计 Hamming distance 和受影响符号
    size_t total_hamming = 0;
    size_t symbol_mismatch_count = 0;
    std::vector<size_t> error_indices;

    for (size_t i = 0; i < total_elements; ++i) {
        uint64_t clean = h_clean_ntt.get()[i];
        uint64_t fault = h_faulty_ntt.get()[i];
        uint64_t diff  = clean ^ fault;
        if (diff != 0) {
            ++symbol_mismatch_count;
            if (symbol_mismatch_count <= 128) {
                error_indices.push_back(i);
            }
        }
        total_hamming += static_cast<size_t>(__builtin_popcountll(diff));
    }

    if (total_hamming != 0) {
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

    if (symbol_mismatch_count > 0 && symbol_mismatch_count <= 128) {
        std::cout << "[Debug] Symbol mismatch indices:\n";
        for (size_t idx : error_indices) {
            std::cout << "  - Index " << idx << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <log_dim> <batch_size> <num_flips> <num_target_symbols>\n";
        return 1;
    }

    size_t log_dim            = static_cast<size_t>(std::stoul(argv[1]));
    size_t batch_size         = static_cast<size_t>(std::stoul(argv[2]));
    int num_flips             = std::stoi(argv[3]);
    int num_target_symbols    = std::stoi(argv[4]);

    if (num_flips < 1 || num_target_symbols < 1) {
        std::cerr << "Error: <num_flips> and <num_target_symbols> must be >= 1\n";
        return 1;
    }
        
    test_nwt(log_dim, batch_size, num_flips, num_target_symbols);
    return 0;
}
