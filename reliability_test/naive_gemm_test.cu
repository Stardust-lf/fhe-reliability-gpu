// bgv_gemm_key_reuse.cpp
// Refactored BGV dot-product to reuse keys instead of regenerating on each call.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "evaluate.cuh"    // multiply, add_inplace, rotate_inplace
#include "example.h"       // print_vector, print_parameters
#include "galois.cuh"      // PhantomGaloisTool, PhantomGaloisKey
#include "phantom.h"
#include "util.cuh"        // PhantomBatchEncoder, PhantomPlaintext, PhantomCiphertext

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

// Single encrypted dot-product using pre-generated keys.
vector<uint64_t> encrypted_dot_product(
    EncryptionParameters &parms,
    PhantomContext &context,
    PhantomSecretKey &secret_key,
    PhantomPublicKey &public_key,
    PhantomRelinKey &relin_keys,
    PhantomGaloisKey &gal_keys,
    PhantomBatchEncoder &batch_encoder)
{
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size   = slot_count / 2;

    // 1) Generate random inputs
    vector<uint64_t> input1(slot_count), input2(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        input1[i] = rand() % 100;
        input2[i] = rand() % 100;
    }

    // 2) Encode & encrypt
    PhantomPlaintext x_plain = batch_encoder.encode(context, input1);
    PhantomPlaintext y_plain = batch_encoder.encode(context, input2);
    PhantomCiphertext x_cipher, y_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    // 3) Multiply + relinearize + mod-switch
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace(context, xy_cipher, relin_keys);
    mod_switch_to_next_inplace(context, xy_cipher);

    // 4) Sum all slots via binary-tree rotations
    for (size_t step = 1; step < row_size; step <<= 1) {
        PhantomCiphertext tmp = xy_cipher;
        rotate_inplace(context, tmp, step, gal_keys);
        add_inplace(context, xy_cipher, tmp);
    }

    // 5) Decrypt & decode
    PhantomPlaintext result_plain = secret_key.decrypt(context, xy_cipher);
    return batch_encoder.decode(context, result_plain);
}

int main() {
    srand(time(nullptr));

    // 1) BGV parameters
    EncryptionParameters parms(scheme_type::bgv);
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        CoeffModulus::Create(poly_modulus_degree, {50, 50, 50, 50, 50, 50})
    );
    parms.set_special_modulus_size(2);
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

    PhantomContext context(parms);
    print_parameters(context);
    cout << endl;

    // 2) Generate keys & encoder once
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys  = secret_key.gen_relinkey(context);
    PhantomGaloisKey gal_keys    = secret_key.create_galois_keys(context);
    PhantomBatchEncoder batch_encoder(context);

    // 3) Run dot-product many times, reusing keys
    for (int i = 0; i < 100; ++i) {
        vector<uint64_t> result = encrypted_dot_product(
            parms, context,
            secret_key, public_key,
            relin_keys, gal_keys,
            batch_encoder
        );
        // Optional: print or verify `result`
        // cout << "Run " << i << " result[0]=" << result[0] << endl;
    }

    return 0;
}
