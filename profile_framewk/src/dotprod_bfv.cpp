#include "examples.h"
#include <seal/seal.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>

using namespace std;
using namespace seal;

// ----------------------------------------------------------------------------
// 封装：Hadamard 乘 + rotate/add 归约，返回密文中所有槽的和落在每个槽里的点积
// ----------------------------------------------------------------------------

Ciphertext compute_dot_product(
    Evaluator        &evaluator,
    const Ciphertext &enc1,
    const Ciphertext &enc2,
    const RelinKeys  &relin_keys,
    const GaloisKeys &gal_keys,
    size_t            slot_count)
{
    std::cout << "function: DOTPRODUCT" << std::endl;
    chrono::high_resolution_clock::time_point time_start, time_end, time_start_func, time_end_func;
    chrono::microseconds time_diff;
    time_start = chrono::high_resolution_clock::now();
    // 1. Hadamard multiply + relinearize
    Ciphertext prod;
    std::cout << "frontend: MULTIPLY_BFV" << std::endl;
    time_start = chrono::high_resolution_clock::now();
    time_start_func = chrono::high_resolution_clock::now();
    evaluator.multiply(enc1, enc2, prod);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "frontend: MULTIPLY_BFV[" << time_diff.count() << " microseconds]" << endl;
    std::cout << "frontend: RELIN" << std::endl;
    time_start = chrono::high_resolution_clock::now();
    evaluator.relinearize_inplace(prod, relin_keys);
    time_end = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
    cout << "frontend: RELIN[" << time_diff.count() << " microseconds]" << endl;

    // 2. Row-wise reduction
    for (size_t step = 1; step < slot_count / 2; step <<= 1)
    {
        Ciphertext tmp;
        std::cout << "frontend: ROTATE" << std::endl;
        time_start = chrono::high_resolution_clock::now();
        evaluator.rotate_rows(prod, static_cast<int>(step), gal_keys, tmp);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        cout << "frontend: ROTATE[" << time_diff.count() << " microseconds]" << endl;
        evaluator.add_inplace(prod, tmp);
    }
    evaluator.mod_switch_to_next_inplace(prod);
    // 3. Column-wise reduction
    {
        Ciphertext tmp;
        std::cout << "frontend: ROTATE" << std::endl;
        time_start = chrono::high_resolution_clock::now();
        evaluator.rotate_columns(prod, gal_keys, tmp);
        time_end = chrono::high_resolution_clock::now();
        time_diff = chrono::duration_cast<chrono::microseconds>(time_end - time_start);
        cout << "frontend: ROTATE[" << time_diff.count() << " microseconds]" << endl;
        evaluator.add_inplace(prod, tmp);
    }

    time_end_func = chrono::high_resolution_clock::now();
    time_diff = chrono::duration_cast<chrono::microseconds>(time_end_func - time_start_func);
    cout << "function: DOTPRODUCT[" << time_diff.count() << " microseconds]" << endl;
    return prod;
}

// ----------------------------------------------------------------------------
// 主函数：包含随机向量生成、离线点积、同态点积、解密与验证
// ----------------------------------------------------------------------------
int main()
{
    // —— 1. SEAL 参数 & 上下文 —— 
    EncryptionParameters parms(scheme_type::bfv);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_plain_modulus(786433);  // 支持批量的素数
    parms.set_coeff_modulus(
        CoeffModulus::Create(poly_modulus_degree, {60, 60, 40})
    );
    // size_t poly_modulus_degree = 4096;  // N = 2^17 = 131072
    // parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(
    //     CoeffModulus::Create(poly_modulus_degree, {60, 60, 60, 60, 60})
    // );
    print_parameters(parms);

    SEALContext context(parms);
    if (!context.first_context_data()->qualifiers().using_batching)
    {
        cerr << "ERROR: Batching not supported with these parameters." << endl;
        return 1;
    }

    // —— 2. 密钥生成 —— 
    KeyGenerator keygen(context);
    SecretKey  secret_key = keygen.secret_key();
    PublicKey  public_key;
    keygen.create_public_key(public_key);
    RelinKeys  relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);

    // —— 3. 编/解/评 & 编码器 —— 
    Encryptor     encryptor(context, public_key);
    Decryptor     decryptor(context, secret_key);
    Evaluator     evaluator(context);
    BatchEncoder  batch_encoder(context);

    // —— 4. 随机向量 & 离线点积 —— 
    size_t slot_count = batch_encoder.slot_count();  // = poly_modulus_degree
    mt19937_64 gen(random_device{}());
    uniform_int_distribution<uint64_t> dist(0, 15);  // 例如取 [0,15] 范围

    vector<uint64_t> vec1(slot_count), vec2(slot_count);
    for (size_t i = 0; i < slot_count; i++)
    {
        vec1[i] = dist(gen);
        vec2[i] = dist(gen);
    }

    // 离线计算（完整整数和，无取模）
    uint64_t expected_dot = inner_product(vec1.begin(), vec1.end(), vec2.begin(), uint64_t(0));

    // —— 5. 编码 & 加密 —— 
    Plaintext p1, p2;
    batch_encoder.encode(vec1, p1);
    batch_encoder.encode(vec2, p2);

    Ciphertext c1, c2;
    encryptor.encrypt(p1, c1);
    encryptor.encrypt(p2, c2);

    // —— 6. 同态点积 & 收集 Callgrind stats —— 
    Ciphertext prod = compute_dot_product(
        evaluator, c1, c2, relin_keys, gal_keys, slot_count);

    // —— 7. 解密 & 解码 —— 
    Plaintext plain_result;
    decryptor.decrypt(prod, plain_result);
    vector<uint64_t> result;
    batch_encoder.decode(plain_result, result);

    // —— 8. 输出 & 验证 —— 
    uint64_t encrypted_dot = result[0];
    cout << "Encrypted dot product (slot 0) = " << encrypted_dot << endl;
    cout << "Expected dot product           = " << expected_dot << endl;
    if (encrypted_dot == expected_dot)
        cout << "✔ Verification passed." << endl;
    else
        cout << "✘ Verification failed!" << endl;

    return 0;
}
