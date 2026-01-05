#include "seal/seal.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
using namespace seal;

int main() {
    // 1. 设置加密参数
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 60, 40, 40, 60 }));

    SEALContext context(parms);
    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    Encryptor encryptor(context, keygen.public_key());
    Decryptor decryptor(context, secret_key);
    Evaluator evaluator(context);
    CKKSEncoder encoder(context);

    size_t slot_count = encoder.slot_count();
    double scale = pow(2.0, 40);

    // 2. 准备明文向量: [1.0, 0, 0, ...]
    vector<double> input_pt(slot_count, 0.0);
    input_pt[0] = 1.0;
    Plaintext plain;
    encoder.encode(input_pt, scale, plain);

    // ==========================================
    // 3. 手动注入噪声：在多项式环级别修改系数
    // 在 SEAL 中，Plaintext 的 data() 返回的是系数数组。
    // 我们给第一个多项式的第一个系数手动 +1 (模拟攻击/噪声)
    cout << ">>> 正在明文多项式中注入大小为 1 的手动噪声..." << endl;
    plain.data()[0] += 1; 
    // ==========================================

    // 4. 准备密文向量: [1.0, 0, 0, ...]
    vector<double> input_ct(slot_count, 0.0);
    input_ct[0] = 1.0;
    Plaintext plain_for_ct;
    encoder.encode(input_ct, scale, plain_for_ct);
    
    Ciphertext encrypted;
    encryptor.encrypt(plain_for_ct, encrypted);

    // 5. 执行明密乘法: res = encrypted * plain (含噪声的明文)
    Ciphertext result_encrypted;
    evaluator.multiply_plain(encrypted, plain, result_encrypted);

    // 6. 解密并解码
    Plaintext plain_result;
    decryptor.decrypt(result_encrypted, plain_result);
    vector<double> output;
    encoder.decode(plain_result, output);

    // 7. 打印结果 (观察噪声扩散)
    cout << "\n解密后的前 5 个 Slot 结果:" << endl;
    for (size_t i = 0; i < 5; i++) {
        printf("Slot %zu: %.10f\n", i, output[i]);
    }

    // 8. 统计分析
    int wrong_count = 0;
    for (double val : output) {
        // 预期的 element-wise 结果应该是 [1, 0, 0...]
        // 我们统计偏离程度较大的 Slot
        if (abs(val) > 0.001 && abs(val - 1.0) > 0.001) {
            wrong_count++;
        }
    }
    printf("\n--- 实验统计 ---");
    printf("\n总 Slots 数: %zu", slot_count);
    printf("\n受噪声/攻击影响严重的 Slots 数: %d\n", wrong_count);

    return 0;
}
