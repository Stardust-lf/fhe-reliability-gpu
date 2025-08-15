#include "examples.h"
#include <valgrind/callgrind.h>
#include "seal/galoiskeys.h"
#include "seal/modulus.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <seal/seal.h>
#include <vector>

using namespace std;
using namespace seal;

// ----------------------------------------------------------------------------
// 封装：Hadamard 乘 + rotate 折叠，返回槽 0 上的点积
// ----------------------------------------------------------------------------
Ciphertext compute_dot_product(Evaluator &evaluator, const Ciphertext &enc1,
                               const Ciphertext &enc2,
                               const RelinKeys &relin_keys,
                               const GaloisKeys &gal_keys, size_t slot_count) {
    std::cout << "function: DOTPRODUCT" << std::endl;
    chrono::high_resolution_clock::time_point time_start, time_end,
        time_start_func, time_end_func;
    chrono::microseconds time_diff;
    Ciphertext prod;
    std::cout << "frontend: MULTIPLY_CKKS" << std::endl;
    time_start = chrono::high_resolution_clock::now();
    time_start_func = chrono::high_resolution_clock::now();

    evaluator.multiply(enc1, enc2, prod);
    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "frontend: MULTIPLY_CKKS[" << time_diff.count() << " microseconds]"
         << endl;

    std::cout << "frontend: RELIN" << std::endl;
    time_start = chrono::high_resolution_clock::now();
    evaluator.relinearize_inplace(prod, relin_keys);
    time_end = chrono::high_resolution_clock::now();
    time_diff =
        chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "frontend: RELIN[" << time_diff.count() << " microseconds]" << endl;

    // 2. 轮询折叠（fold）：每次旋转并累加
    for (size_t step = 1; step < slot_count; step <<= 1) {
        Ciphertext tmp;
        std::cout << "frontend: ROTATE" << std::endl;
        time_start = chrono::high_resolution_clock::now();
        evaluator.rotate_vector(prod, static_cast<int>(step), gal_keys, tmp);
        time_end = chrono::high_resolution_clock::now();
        time_diff =
            chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        cout << "frontend: ROTATE[" << time_diff.count() << " microseconds]"
             << endl;
        evaluator.add_inplace(prod, tmp);
    }
    time_end_func = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end_func -
                                                            time_start_func);
    cout << "function: DOTPRODUCT[" << time_diff.count() << " microseconds]"
         << endl;
    return prod;
}

vector<int> parse_coeff_bits(const string &s) {
    vector<int> bits;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ',')) {
        try {
            bits.push_back(stoi(item));
        } catch (...) {
            cerr << "Invalid coefficient bit size: " << item << endl;
            exit(1);
        }
    }
    return bits;
}

int main(int argc, char *argv[]) {
    // 1. CKKS 参数 & 上下文
    CALLGRIND_STOP_INSTRUMENTATION;
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 32768;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_plain_modulus(786433);  // 支持批量的素数
    // std::vector<int> modulus_bits(24, 36);
    // parms.set_coeff_modulus(CoeffModulus::Create(
    //     poly_modulus_degree,
    //     {37, 37, 37, 37}));
    //     CoeffModulus::Create(
    parms.set_coeff_modulus(
    CoeffModulus::Create(
        poly_modulus_degree,
        {
            31,31,31,31,31,31,31,31,31,31,   // 10 × 31 位
            30,30,30,30,30,30,30,30,30,30,30  // 11 × 30 位
        }
        )
    );

    // parms.set_coeff_modulus(
    //     CoeffModulus::Create(poly_modulus_degree, modulus_bits));

    print_parameters(parms);
    cout << "Plain modulus: " << parms.plain_modulus().value() << endl;
    SEALContext context(parms);
    if (!context.parameters_set()) {
        // 获取错误码（枚举名称）
        auto error_name = context.parameter_error_name();
        // 获取可读的错误信息
        auto error_msg = context.parameter_error_message();

        cerr << "Invalid EncryptionParameters (" << error_name
             << "): " << error_msg << endl;
        return 1;
    }

    // 2. 密钥生成
    KeyGenerator keygen(context);
    SecretKey secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);

    // 指定分解位宽
    // int decomposition_bit_count = 6;

    // 用返回值接口生成带 decomposition_bit_count 的 RelinKeys 和 GaloisKeys
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);

    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    // 3. 编/解/评 & 编码器
    Encryptor encryptor(context, public_key);
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    // 4. 随机浮点向量 & 离线点积（使用 float 类型生成）
    size_t slot_count = encoder.slot_count(); // = poly_modulus_degree/2
    mt19937_64 gen(random_device{}());
    uniform_real_distribution<float> dist(0.0f, 15.0f);

    vector<float> raw1(slot_count), raw2(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        raw1[i] = dist(gen);
        raw2[i] = dist(gen);
    }
    // 离线计算点积（float）
    float expected_dot_f =
        inner_product(raw1.begin(), raw1.end(), raw2.begin(), 0.0f);

    // 转为 double 向量用于 CKKS 编码
    vector<double> vec1(slot_count), vec2(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        vec1[i] = static_cast<double>(raw1[i]);
        vec2[i] = static_cast<double>(raw2[i]);
    }
    double expected_dot = static_cast<double>(expected_dot_f);

    // 5. 编码 & 加密
    double scale = pow(2.0, 40);
    Plaintext p1, p2;
    encoder.encode(vec1, scale, p1);
    encoder.encode(vec2, scale, p2);

    Ciphertext c1, c2;
    encryptor.encrypt(p1, c1);
    encryptor.encrypt(p2, c2);

    // 6. 同态点积
    auto start = chrono::high_resolution_clock::now();
    CALLGRIND_START_INSTRUMENTATION;
    Ciphertext enc_result = compute_dot_product(evaluator, c1, c2, relin_keys,
                                                gal_keys, slot_count);
    CALLGRIND_STOP_INSTRUMENTATION;
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Encrypted dot product time: " << duration.count()
         << " microseconds" << endl;

    // 7. 解密 & 解码
    Plaintext plain_result;
    decryptor.decrypt(enc_result, plain_result);
    vector<double> result;
    encoder.decode(plain_result, result);

    // 8. 输出差异
    double encrypted_dot = result[0];
    double diff = encrypted_dot - expected_dot;
    cerr << fixed << setprecision(6);
    cerr << "Encrypted dot (slot 0) = " << encrypted_dot << endl;
    cerr << "Expected dot           = " << expected_dot << endl;
    cerr << "Difference (enc - exp) = " << diff << endl;

    return 0;
}
