#!/bin/bash

# build for a specific target with cross compilation

# sop on error
set -e

target=$1
if [ -z "$target" ]
then
    echo "Usage: $0 <target_architecture>"
    exit 1
elif [ "$target" == "aarch64-unknown-linux-gnu" ]
then
    linker="aarch64-linux-gnu-gcc"
# elif [ "$target" == "armv7-unknown-linux-gnueabihf" ]
# then
#     linker="arm-linux-gnueabihf-gcc"
else
    echo "Unsupported target: ${target}"
    exit 1
fi
echo "Building release build to target: ${target_profile}"

rustup target add ${target}
linker_variable=$(echo "CARGO_TARGET_$(echo "${target^^}" | sed "s/-/_/g")_LINKER")
export ${linker_variable}=${linker}

mkdir -p ${target}

rm -f ${target}/onnx-native-test
rm -f ${target}/wasmtime-test

echo "Compiling the native ONNX runtime test program"
cd ../native
cargo build --profile release --target=${target}
cp target/${target}/release/onnx-native-test ../bin/${target}

echo "Compiling the wasmtime test program"
cd ../wasm/wasmtime-test
cargo build --profile release --target=${target}
cp target/${target}/release/wasmtime-test ../../bin/${target}

cd ../../bin
