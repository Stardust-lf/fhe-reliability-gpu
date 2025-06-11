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

#include "evaluate.cuh"   // for add_inplace, multiply, rotate, etc.
#include "example.h"
#include "galois.cuh"     // for PhantomGaloisTool
#include "phantom.h"
#include "util.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util; // for PhantomGaloisTool, PhantomGaloisKey

// ------------------------------------------------------------------
// 全局 EncryptionParameters
// ------------------------------------------------------------------
EncryptionParameters parms(scheme_type::bgv);

// ------------------------------------------------------------------
// 1-thread CUDA kernel: flip one bit in data[idx]
// ------------------------------------------------------------------
__global__ void _flip_bit_kernel(uint64_t* data, size_t idx, size_t bit) {
    data[idx] ^= (1ULL << bit);
}

// ------------------------------------------------------------------
// Inject specified number of bit-flips across specified number of symbols
// ------------------------------------------------------------------
void inject_bitflip_ciphertext(PhantomCiphertext &ct,
                               int bits_per_symbol,
                               int num_symbols) {
    // total number of uint64_t coefficients in the ciphertext
    size_t total = ct.size()
                 * ct.coeff_modulus_size()
                 * ct.poly_modulus_degree();

    uint64_t *dptr = ct.data();     // raw device pointer to all coeffs

    // Flip bits_per_symbol bits in each of num_symbols symbols
    for (int s = 0; s < num_symbols; ++s) {
        // pick a random symbol index
        size_t idx = rand() % total;
        for (int b = 0; b < bits_per_symbol; ++b) {
            size_t bit = rand() % 64;
            // launch kernel to flip this bit
            _flip_bit_kernel<<<1,1>>>(dptr, idx, bit);
            cudaDeviceSynchronize();
            cerr << "Injected bitflip @ idx=" << idx
                 << ", bit=" << bit << "\n";
        }
    }
}

// ------------------------------------------------------------------
// Main test: encrypt, inject fault, homomorphic multiply+rotate-sum,
// decrypt, decode, then CPU check.
// ------------------------------------------------------------------
void dot_product_test(PhantomContext &context,
                      int bits_per_symbol,
                      int num_symbols) {
    cout << "Example: BGV HomMul test" << endl;

    // Key generation
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    // Batch encoder
    PhantomBatchEncoder batch_encoder(context);
    size_t slot_count = batch_encoder.slot_count();
    size_t row_size   = slot_count / 2;
    cout << "Plaintext matrix row size: " << row_size << endl;

    // 1) Generate two random input vectors
    vector<uint64_t> input1(slot_count), input2(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        input1[i] = rand() % 100;
        input2[i] = rand() % 100;
    }
    cout << "Input vector 1: "; print_vector(input1, 3, 7);
    cout << "Input vector 2: "; print_vector(input2, 3, 7);

    // CPU baseline elementwise
    uint64_t mod = parms.plain_modulus().value();
    vector<uint64_t> baseline(slot_count);
    for (size_t i = 0; i < slot_count; ++i) {
        baseline[i] = (input1[i] * input2[i]) % mod;
    }

    // 2) Encode plaintexts
    PhantomPlaintext x_plain = batch_encoder.encode(context, input1);
    PhantomPlaintext y_plain = batch_encoder.encode(context, input2);

    // 3) Encrypt
    PhantomCiphertext x_cipher, y_cipher;
    public_key.encrypt_asymmetric(context, x_plain, x_cipher);
    public_key.encrypt_asymmetric(context, y_plain, y_cipher);

    // 4) Inject bit-flip fault into x_cipher
    inject_bitflip_ciphertext(x_cipher, bits_per_symbol, num_symbols);

    // 5) Homomorphic element-wise multiplication x * y
    cout << "Compute x * y homomorphically..." << endl;
    PhantomCiphertext xy_cipher = multiply(context, x_cipher, y_cipher);
    relinearize_inplace(context, xy_cipher, relin_keys);
    mod_switch_to_next_inplace(context, xy_cipher);

    // 6) Immediate decrypt+decode to inspect and compare elementwise
    {
        PhantomPlaintext xy_plain = secret_key.decrypt(context, xy_cipher);
        vector<uint64_t> prod = batch_encoder.decode(context, xy_plain);

        size_t symbol_errors = 0;
        size_t bit_errors    = 0;
        for (size_t i = 0; i < slot_count; ++i) {
            uint64_t a = baseline[i];
            uint64_t b = prod[i];
            if (a != b) {
                ++symbol_errors;
                uint64_t diff = a ^ b;
                bit_errors += __builtin_popcountll(diff);
            }
        }

        cout << "Raw product vector: "; print_vector(prod, 3, 7);
        cout << "CPU baseline      : "; print_vector(baseline, 3, 7);
        cout << "Elementwise symbol errors: "
             << symbol_errors << " / " << slot_count << endl;
        cout << "Elementwise Hamming distance (bit errors): "
             << bit_errors << endl;
    }

    // 7) Rotate-and-add to compute the dot product
    PhantomGaloisKey gal_keys = secret_key.create_galois_keys(context);
    for (size_t step = 1; step < row_size; step <<= 1) {
        PhantomCiphertext rotated = xy_cipher;
        rotate_inplace(context, rotated, step, gal_keys);
        add_inplace(context, xy_cipher, rotated);
    }

    // 8) Final decrypt + decode → dot product
    PhantomPlaintext dp_plain = secret_key.decrypt(context, xy_cipher);
    vector<uint64_t> result = batch_encoder.decode(context, dp_plain);
    uint64_t result_full = (result[0] + result[row_size]) % mod;

    // 9) CPU baseline dot-product
    uint64_t expected = 0;
    for (size_t i = 0; i < slot_count; ++i) {
        expected = (expected + baseline[i]) % mod;
    }

    // 10) Print & compare with Hamming and percentage error
    uint64_t diff_dp = result_full ^ expected;
    size_t dp_bit_errors = __builtin_popcountll(diff_dp);
    cout << "Decrypted dot product = " << result_full << endl;
    cout << "Expected (CPU)         = " << expected     << endl;
    cout << "Dot product bit errors (Hamming distance): "
         << dp_bit_errors << endl;
    uint64_t abs_diff = result_full > expected
                       ? result_full - expected
                       : expected - result_full;
    cout << "Absolute difference   = " << abs_diff << endl;
    if (expected != 0) {
        double pct_err = static_cast<double>(abs_diff)
                       / static_cast<double>(expected)
                       * 100.0;
        cout << "Percentage error      = " << pct_err << "%" << endl;
    } else {
        cout << "Percentage error      = undefined (expected is zero)" << endl;
    }
    if (result_full == expected) {
        cout << "✔ Dot product matches CPU result." << endl;
    } else {
        cout << "✖ MISMATCH detected!" << endl;
    }
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc != 3) {
        cerr << "Usage: " << argv[0]
             << " <bits_per_symbol> <num_symbols>\n";
        return 1;
    }
    int bits_per_symbol = atoi(argv[1]);
    int num_symbols    = atoi(argv[2]);

    // Configure global parms
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(
        CoeffModulus::Create(poly_modulus_degree, {50,50,50,50,50,50}));
    parms.set_special_modulus_size(2);
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));

    // Build context once
    PhantomContext context(parms);

    // Print parameters
    print_parameters(context);
    cout << endl;

    // Run the test with injection parameters
    dot_product_test(context, bits_per_symbol, num_symbols);

    return 0;
}
