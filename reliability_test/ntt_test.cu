#include "phantom.h"
#include <iostream>
#include <random>
#include <memory>
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


void test_nwt(size_t log_dim, size_t batch_size, int bitpos) {
    phantom::util::cuda_stream_wrapper stream_wrapper;
    const auto &s = stream_wrapper.get_stream();
    size_t dim = 1ULL << log_dim;  // N = 2^log_dim

    // 1. Host 端生成模数列表
    const auto h_modulus = CoeffModulus::Create(dim, std::vector<int>(batch_size, 50));

    // 2. 复制到 Device 端的 DModulus 数组
    auto modulus = phantom::util::make_cuda_auto_ptr<DModulus>(batch_size, s);
    for (size_t i = 0; i < batch_size; ++i) {
        modulus.get()[i].set(h_modulus[i].value(),
                             h_modulus[i].const_ratio()[0],
                             h_modulus[i].const_ratio()[1]);
    }

    // 3. 构建并初始化 GPU 端的 DNTTTable（只需要正向 Twiddle）
    DNTTTable d_ntt_tables;
    d_ntt_tables.init(dim, batch_size, s);
    for (size_t i = 0; i < batch_size; ++i) {
        NTT h_ntt_table(log_dim, h_modulus[i]);
        d_ntt_tables.set(&modulus.get()[i],
                         h_ntt_table.get_from_root_powers().data(),
                         h_ntt_table.get_from_root_powers_shoup().data(),
                         /*逆向 Twiddle*/ nullptr, nullptr,
                         /*n_inv*/ 0, 0,
                         i, s);
    }

    // 4. 初始化随机输入，Host 端
    auto h_data = std::make_unique<uint64_t[]>(batch_size * dim);
    std::random_device rd;
    std::mt19937_64 rng(rd());
    for (size_t i = 0; i < batch_size; ++i) {
        uint64_t mod = h_modulus[i].value();
        std::uniform_int_distribution<uint64_t> dist(0, mod - 1);
        for (size_t j = 0; j < dim; ++j) {
            h_data.get()[i * dim + j] = dist(rng);
        }
    }

    // 5. 准备存储 clean 和 faulty 的 NTT 输出
    auto h_clean_ntt  = std::make_unique<uint64_t[]>(batch_size * dim);
    auto h_faulty_ntt = std::make_unique<uint64_t[]>(batch_size * dim);

    // 6. 将干净输入拷贝到 Device 并执行 2D NTT（只是前向部分）
    auto d_data = phantom::util::make_cuda_auto_ptr<uint64_t>(batch_size * dim, s);
    cudaMemcpyAsync(d_data.get(),
                    h_data.get(),
                    batch_size * dim * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, s);
    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, /*start_mod=*/0, s);
    cudaMemcpyAsync(h_clean_ntt.get(),
                    d_data.get(),
                    batch_size * dim * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // 7. 在 Host 端对 h_data 做“单点随机比特翻转”
    // std::uniform_int_distribution<size_t> idx_dist(0, batch_size * dim - 1);
    // size_t flip_idx = idx_dist(rng);
    // uint64_t mask = (1ULL << bitpos);
    // h_data.get()[flip_idx] ^= mask;
    // std::cout << "[2D] Flipped bit " << bitpos
    //           << " of element index " << flip_idx
    //           << " (mask = 0x" << std::hex << mask << std::dec << ")\n";

    // 8. 翻转后拷贝到 Device 并再执行 2D NTT
    cudaMemcpyAsync(d_data.get(),
                    h_data.get(),
                    batch_size * dim * sizeof(uint64_t),
                    cudaMemcpyHostToDevice, s);
    nwt_2d_radix8_forward_inplace(d_data.get(), d_ntt_tables, batch_size, /*start_mod=*/0, s);
    cudaMemcpyAsync(h_faulty_ntt.get(),
                    d_data.get(),
                    batch_size * dim * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    // 9. 比较 clean_ntt vs faulty_ntt，统计 mismatch 及 bit-level Hamming distance
    size_t total_hamming = 0;
    std::cout << "[2D] Comparing clean vs faulty NTT outputs...\n";
    for (size_t i = 0; i < batch_size * dim; ++i) {
        uint64_t clean = h_clean_ntt.get()[i];
        uint64_t fault = h_faulty_ntt.get()[i];
        uint64_t diff = clean ^ fault;
        total_hamming += static_cast<size_t>(__builtin_popcountll(diff));
    }
    if(total_hamming != 0){
        std::cout << "ERROR! Total bitwise Hamming distance = "
                  << total_hamming << "\n";
    }else{
        std::cout <<"ALL CORRECT" << std::endl;
    }
    // std::cout << "[2D] test_nwt_2d (log_dim=" << log_dim
    //           << ", batch_size=" << batch_size << ") done\n\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <log_dim> <batch_size> <bitpos>\n";
        return 1;
    }

    size_t log_dim = static_cast<size_t>(std::stoul(argv[1]));
    size_t batch_size = static_cast<size_t>(std::stoul(argv[2]));
    int bitpos = std::stoi(argv[3]);
    test_nwt(log_dim, batch_size, bitpos);

    return 0;
}
