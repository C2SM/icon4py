#!/bin/bash

# Ensure the script exits on the first error
set -e

# Step 1: Compile the C wrapper with OpenACC support
nvc -c acc_wrapper.c -o acc_wrapper.o -acc -Minfo=acc


nvfortran -cpp -I. -Wl,-rpath=. -L. \
    identity_plugin.f90 \
    test_gpu.f90 \
    acc_wrapper.o \
    -lidentity_plugin \
    -o identity_plugin -acc -Minfo=acc

echo "Compilation successful."
