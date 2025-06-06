#!/bin/bash

nvcc -std=c++17 \
    -I../phantom-fhe/include \
    -L../phantom-fhe/build/lib -lPhantom \
    ntt_test.cu -o ntt_test

