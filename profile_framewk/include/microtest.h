#pragma once

#include "examples.h"
#include "seal/encryptionparams.h"
#include <valgrind/callgrind.h>
#include <iostream>
#include <seal/util/globals.h>
#include <seal/seal.h>
#include <seal/seal.h>
#include <string>

using namespace std;
using namespace seal;

namespace microtest {
enum class SchemeType { BGV, BFV, CKKS };
class MicroTest {
public:
    // 1a) A private default ctor that gives `ctx` & `context_` *some* initial values.
    MicroTest()
        : ctx(EncryptionParameters(scheme_type::bfv)),         // pick any scheme
          context_(ctx)                   // build a valid SEALContext
    {}

    // 1b) Your real public ctor just delegates to the above, then re-inits.
    explicit MicroTest(SchemeType scheme)
        : MicroTest()           // calls the default ctor first
    {
        init(scheme);          // now override `ctx` & `context_` properly
    }

    virtual ~MicroTest() = default;
    virtual std::string name()  const = 0;
    virtual void        run()         = 0;

protected:
    EncryptionParameters ctx;
    SEALContext        context_;

    void init(SchemeType scheme) {
        switch (scheme) {
        case SchemeType::BGV:  create_bgv_context();  break;
        case SchemeType::BFV:  create_bfv_context();  break;
        case SchemeType::CKKS: create_ckks_context(); break;
        }
    }

    void create_bgv_context()
    {
        cout << "BGV Performance Test…\n";

        // 1) 先构造带 scheme 的 EncryptionParameters  
        ctx = EncryptionParameters(scheme_type::bgv);

        // 2) 然后在它上面设参数  
        constexpr size_t DEG = 4096;
        ctx.set_poly_modulus_degree(DEG);
        ctx.set_coeff_modulus(CoeffModulus::BFVDefault(DEG));
        ctx.set_plain_modulus(786433);
    }

    void create_bfv_context()
    {
        cout << "BFV Performance Test…\n";
        ctx = EncryptionParameters(scheme_type::bfv);
        constexpr size_t DEG = 4096;
        ctx.set_poly_modulus_degree(DEG);
        ctx.set_coeff_modulus(CoeffModulus::BFVDefault(DEG));
        ctx.set_plain_modulus(786433);
    }

    void create_ckks_context()
    {
        cout << "CKKS Performance Test…\n";

        ctx = EncryptionParameters(scheme_type::ckks);
        constexpr size_t DEG = 4096;
        ctx.set_poly_modulus_degree(DEG);
        ctx.set_coeff_modulus(CoeffModulus::BFVDefault(DEG));
        ctx.set_plain_modulus(786433);
    }

};

} // namespace microtest
