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

rustup target add wasm32-wasi

rm -f onnx-native-test
rm -f wasmtime-test
rm -f wasi-nn-onnx-test.wasm
rm -f wasi-nn-onnx-test.wasm.SERIALIZED
rm -f simple-onnx.wasm
rm -f simple-onnx.wasm.SERIALIZED
rm -f *.so*

echo "Compiling the native ONNX runtime test program"
cd ../native
cargo build --profile ${target_profile}
cd ../bin
cp ../native/target/${build_folder}/onnx-native-test ../bin
for lib in $(ls ../native/target/${build_folder}/*.so*)
do
    ln -s ${lib} .
done

# export ORT_DYLIB_PATH="$(pwd)/target/release/libonnxruntime.so"
# on Windows use the following export instead
# export ORT_DYLIB_PATH="$(pwd)/target/release/onnxruntime.dll"

echo "Compiling the wasi-nn Wasm module"
cd ../wasm/wasi-nn
cargo build --profile ${target_profile} --target=wasm32-wasi
cp target/wasm32-wasi/${build_folder}/wasi-nn-onnx-test.wasm ../../bin

echo "Compiling the simple, non wasi-nn, Wasm module"
cd ../simple-onnx
cargo build --profile ${target_profile} --target=wasm32-wasi
cp target/wasm32-wasi/${build_folder}/simple-onnx.wasm ../../bin

echo "Compiling the wasmtime test program"
cd ../wasmtime-test
cargo build --profile ${target_profile}
cp target/${build_folder}/wasmtime-test ../../bin

cd ../../bin
