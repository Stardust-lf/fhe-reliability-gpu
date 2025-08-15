#include "examples.h"
#include <seal/seal.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace std;
using namespace seal;

// ----------------------------------------------
// 打包：将 dim×dim 矩阵按“循环对角线”编码成 dim 个 Plaintext
// 注意：这里的 scale 要与向量加密时一致
// ----------------------------------------------
void pack_diagonals(
    const vector<vector<double>> &mat,
    size_t                        dim,
    CKKSEncoder                  &encoder,
    double                        scale,
    vector<Plaintext>            &diags)
{
    size_t slot_count = encoder.slot_count();
    size_t blocks     = slot_count / dim;
    diags.resize(dim);
    for (size_t k = 0; k < dim; k++)
    {
        vector<double> diag(slot_count);
        for (size_t b = 0; b < blocks; b++)
        {
            for (size_t i = 0; i < dim; i++)
            {
                diag[b*dim + i] = mat[i][(i + k) % dim];
            }
        }
        encoder.encode(diag, scale, diags[k]);
    }
}

// ----------------------------------------------------------------------------
// BSGS 同态矩阵×向量：分块旋转 + 逐槽乘 + 累加
// ----------------------------------------------------------------------------
Ciphertext compute_matvec_bsgs(
    Evaluator                   &evaluator,
    const Ciphertext            &enc_vec,
    const vector<Plaintext>     &diags,
    const RelinKeys             &relin_keys,
    const GaloisKeys            &gal_keys,
    size_t                       dim)
{
    // BSGS 参数：块大小 B 和块数 G
    size_t B = static_cast<size_t>(ceil(sqrt(static_cast<double>(dim))));
    size_t G = static_cast<size_t>((dim + B - 1) / B);

    Ciphertext sum;
    bool first = true;

    for (size_t g = 0; g < G; g++)
    {
        // 1) "巨步"旋转：g * B
        Ciphertext c_g;
        if (g == 0)
        {
            c_g = enc_vec;
        }
        else
        {
            evaluator.rotate_vector(enc_vec, int(g * B), gal_keys, c_g);
        }

        // 2) "细步"旋转 + 乘法 + 重缩放 + 累加
        for (size_t b = 0; b < B; b++)
        {
            size_t k = g * B + b;
            if (k >= dim) break;

            // 2.1) 在巨步结果上再旋转 b
            Ciphertext c_gb;
            if (b == 0)
            {
                c_gb = c_g;
            }
            else
            {
                evaluator.rotate_vector(c_g, int(b), gal_keys, c_gb);
            }

            // 2.2) 与第 k 条对角线相乘
            Ciphertext tmp;
            evaluator.multiply_plain(c_gb, diags[k], tmp);
            evaluator.relinearize_inplace(tmp, relin_keys);

            // 2.3) 立即重缩放到下一层
            evaluator.rescale_to_next_inplace(tmp);

            // 2.4) 累加
            if (first)
            {
                sum = std::move(tmp);
                first = false;
            }
            else
            {
                evaluator.mod_switch_to_inplace(sum, tmp.parms_id());
                evaluator.add_inplace(sum, tmp);
            }
        }
    }

    return sum;
}

int main()
{
    // 1) CKKS 参数 & 上下文
    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    // parms.set_coeff_modulus(
    //     CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60})
    // );
    parms.set_coeff_modulus(
        CoeffModulus::Create(
            poly_modulus_degree,
            {
                31,31,31,31,31,31,31,31,31,31,   // 10 × 31 位
                // 30,30,30,30,30,30,30,30,30,30,30  // 11 × 30 位
            }
        )
    );
    print_parameters(parms);
    SEALContext context(parms);

    // 2) 密钥生成
    KeyGenerator keygen(context);
    SecretKey  secret_key = keygen.secret_key();
    PublicKey  public_key;  keygen.create_public_key(public_key);
    RelinKeys  relin_keys;  keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;    keygen.create_galois_keys(gal_keys);

    Encryptor  encryptor(context, public_key);
    Decryptor  decryptor(context, secret_key);
    Evaluator  evaluator(context);
    CKKSEncoder encoder(context);

    // ============ 随机测试 dim×dim 矩阵 × dim 向量 ============
    const size_t dim = 5;
    mt19937_64 rnd(random_device{}());
    uniform_real_distribution<double> dist(-100.0, 100.0);

    // 3) 随机生成矩阵 W 和向量 x
    vector<vector<double>> W(dim, vector<double>(dim));
    vector<double>         x(dim);
    for (size_t i = 0; i < dim; i++)
    {
        x[i] = dist(rnd);
        for (size_t j = 0; j < dim; j++)
            W[i][j] = dist(rnd);
    }

    // 离线计算期望结果
    vector<double> y_plain(dim, 0.0);
    for (size_t i = 0; i < dim; i++)
        for (size_t j = 0; j < dim; j++)
            y_plain[i] += W[i][j] * x[j];

    // 4) 打包并加密向量 x（slot_count 长度，前 dim 有效）
    size_t slot_count = encoder.slot_count();
    vector<double> vx(slot_count, 0.0);
    for (size_t i = 0; i < dim; i++) vx[i] = x[i];

    double scale = pow(2.0, 40);
    Plaintext p_x; encoder.encode(vx, scale, p_x);
    Ciphertext c_x; encryptor.encrypt(p_x, c_x);

    // 5) 打包矩阵对角线（使用相同 scale）
    vector<Plaintext> diags;
    pack_diagonals(W, dim, encoder, scale, diags);

    // 6) BSGS 同态矩阵×向量
    auto t1 = chrono::high_resolution_clock::now();
    Ciphertext c_y = compute_matvec_bsgs(
        evaluator, c_x, diags, relin_keys, gal_keys, dim);
    auto t2 = chrono::high_resolution_clock::now();
    auto tm = chrono::duration_cast<chrono::microseconds>(t2 - t1);
    cout << "Encrypted matmul (BSGS) time: " << tm.count()
         << " microseconds" << endl;

    // 7) 解密 & 解码
    Plaintext p_y; decryptor.decrypt(c_y, p_y);
    vector<double> y_enc;
    encoder.decode(p_y, y_enc);

    // 8) 输出比较
    cout << fixed << setprecision(6);
    for (size_t i = 0; i < dim; i++)
    {
        cerr << "row " << i << ": enc = "
             << y_enc[i] << ", plain = "
             << y_plain[i] << endl;
    }

    return 0;
}
