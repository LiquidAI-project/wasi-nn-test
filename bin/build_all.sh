#!/bin/bash

target_profile=$1
if [ -z "$target_profile" ]
then
    target_profile="release"
fi
if [ "$target_profile" == "dev" ]
then
    build_folder="debug"
else
    build_folder=${target_profile}
fi
echo "Building with profile: ${target_profile}"

wasi_target="wasm32-wasip1"
rustup target add ${wasi_target}

rm -f onnx-native-test
rm -f wasmtime-test
rm -f wasi-nn-onnx-test.wasm
rm -f wasi-nn-onnx-test.wasm.SERIALIZED
rm -f simple-onnx.wasm
rm -f simple-onnx.wasm.SERIALIZED

echo "Compiling the native ONNX runtime test program"
cd ../native
cargo build --profile ${target_profile}
cp target/${build_folder}/onnx-native-test ../bin
# export ORT_DYLIB_PATH="$(pwd)/target/release/libonnxruntime.so"
# on Windows use the following export instead
# export ORT_DYLIB_PATH="$(pwd)/target/release/onnxruntime.dll"

echo "Compiling the wasi-nn Wasm module"
cd ../wasm/wasi-nn
cargo build --profile ${target_profile} --target=${wasi_target}
cp target/${wasi_target}/${build_folder}/wasi-nn-onnx-test.wasm ../../bin

echo "Compiling the simple, non wasi-nn, Wasm module"
cd ../simple-onnx
cargo build --profile ${target_profile} --target=${wasi_target}
cp target/${wasi_target}/${build_folder}/simple-onnx.wasm ../../bin

echo "Compiling the wasmtime test program"
cd ../wasmtime-test
cargo build --profile ${target_profile}
cp target/${build_folder}/wasmtime-test ../../bin

cd ../../bin
