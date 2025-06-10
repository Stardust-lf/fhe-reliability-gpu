// dotprod.cu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>


#include "phantom.h"
#include "util.cuh"
#include "evaluate.cuh"    // for add_inplace, multiply, rotate, etc.
#include "galois.cuh"      // for PhantomGaloisTool
#include "example.h"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;  // for PhantomGaloisTool, PhantomGaloisKey

void dot_product_test(EncryptionParameters &parms, PhantomContext &context) {
    cout << "Example: BGV HomMul test" << endl;

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    // Generate two random plaintext vectors
    vector<uint64_t> input1(slot_count), input2(slot_count);
    // for (size_t i = 0; i < slot_count; i++) {
    //     input1[i] = rand() % parms.plain_modulus().value();
    //     input2[i] = rand() % parms.plain_modulus().value();
    // }
    for (size_t i = 0; i < slot_count; i++) {
        input1[i] = rand() % 100;
        input2[i] = rand() % 100;
    }
    cout << "Input vector 1: " << endl;
    print_vector(input1, 3, 7);
    cout << "Input vector 2: " << endl;
    print_vector(input2, 3, 7);

    PhantomPlaintext x_plain = batch_encoder.encode(context, input1);
    PhantomPlaintext y_plain = batch_encoder.encode(context, input2);

    PhantomCiphertext x_cipher, y_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    // Compute x * y
    cout << "Compute x*y." << endl;
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace(context, xy_cipher, relin_keys);
    mod_switch_to_next_inplace(context, xy_cipher);
    PhantomPlaintext xy_plain = secret_key.decrypt(context, xy_cipher);
    vector<uint64_t> result = batch_encoder.decode(context, xy_plain);
    cout << "Result vector: " << endl;
    print_vector(result, 3, 7);

    for (size_t step = 1; step < slot_count; step <<= 1) {
        PhantomCiphertext rotated;
        PhantomGaloisKey gal_keys = secret_key.gen_galoiskey(context);
        rotate_rows(context, xy_cipher, step, gal_keys, rotated); // 需要伽罗瓦密钥
        add_inplace(context, xy_cipher, rotated);
    }
    PhantomPlaintext dot_plain = secret_key.decrypt(context, xy_cipher);
    vector<uint64_t> result = batch_encoder.decode(context, dot_plain);
    cout << "Dot product result: " << result[0] << endl;

    // {
    //     vector<uint64_t> expected(slot_count);
    //     uint64_t mod = parms.plain_modulus().value();
    //     for (size_t i = 0; i < slot_count; i++) {
    //         uint64_t temp = (input1[i] * input2[i]) % mod;
    //         expected[i] = (temp * input1[i]) % mod;
    //     }
    //     bool ok = true;
    //     for (size_t i = 0; i < slot_count; i++) {
    //         ok &= (result[i] == expected[i]);
    //     }
    //     if (!ok) {
    //         report_mismatch(expected, result);
    //     }
    // }

    // // Homomorphic squaring (x^2)
    // cout << "Example: BGV HomSqr test" << endl;
    // cout << "Message vector: " << endl;
    // // print_vector(input1, 3, 7);

    // PhantomPlaintext xx_plain;
    // batch_encoder.encode(context, input1, x_plain);
    // public_key.encrypt_asymmetric(context, x_plain, x_cipher);

    // cout << "Compute and relinearize x^2." << endl;
    // multiply_inplace(context, x_cipher, x_cipher);
    // relinearize_inplace(context, x_cipher, relin_keys);
    // mod_switch_to_next_inplace(context, x_cipher);

    // secret_key.decrypt(context, x_cipher, xx_plain);
    // batch_encoder.decode(context, xx_plain, result);
    // cout << "Result vector: " << endl;
    // // print_vector(result, 3, 7);
    // {
    //     vector<uint64_t> expected(slot_count);
    //     uint64_t mod = parms.plain_modulus().value();
    //     for (size_t i = 0; i < slot_count; i++) {
    //         expected[i] = (input1[i] * input1[i]) % mod;
    //     }
    //     bool ok = true;
    //     for (size_t i = 0; i < slot_count; i++) {
    //         ok &= (result[i] == expected[i]);
    //     }
    //     if (!ok) {
    //         report_mismatch(expected, result);
    //     }
    // }
    // result.clear();
    // input1.clear();
}

int main(){
    srand(time(NULL));

    EncryptionParameters parms(scheme_type::bgv);

    // Example parameters for BGV
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {50, 50, 50, 50, 50, 50}));
    parms.set_special_modulus_size(2);
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    PhantomContext context(parms);

    print_parameters(context);
    cout << endl;
    dot_product_test(parms, context);
}
