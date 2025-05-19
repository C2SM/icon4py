#!/bin/bash

# Save the current directory
ORIG_DIR=$(pwd)

# Clone the repository
git clone --recursive https://github.com/spcl/EBCC.git

# Enter the cloned repo
cd EBCC || exit 1

# Create and enter the build directory
mkdir -p ./src/build
cd ./src/build || exit 1

# Run cmake and build/install
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 8 && make install

# Return to the original directory
cd "$ORIG_DIR"
