#!/bin/bash

nvcc -std=c++17 \
    -I../phantom-fhe/include \
    -L../phantom-fhe/build/lib -lPhantom \
    ntt_test.cu -o ntt_test

nvcc -std=c++17 \
    -I../phantom-fhe/include \
    -L../phantom-fhe/build/lib -lPhantom \
    ntt_real_test.cu -o ntt_real_test

nvcc -std=c++17 \
    -I../phantom-fhe/include \
    -I../phantom-fhe/examples \
    -L../phantom-fhe/build/lib -lPhantom \
    dotprod.cu -o dotprod
