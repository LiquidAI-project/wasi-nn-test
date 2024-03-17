#!/bin/bash

rustup target add wasm32-wasi

rm -f onnx-native-test
rm -f wasmtime-test
rm -f simple-test
rm -f wasi-nn-onnx-test.wasm
rm -f wasi-nn-onnx-test.wasm.SERIALIZED
rm -f simple-onnx.wasm
rm -f simple-onnx.wasm.SERIALIZED

cd ../native
cargo build --release
cp target/release/onnx-native-test ../bin

cd ../wasm/wasi-nn
cargo build --release --target=wasm32-wasi
cp target/wasm32-wasi/release/wasi-nn-onnx-test.wasm ../../bin

cd ../wasmtime-test
cargo build --release
cp target/release/wasmtime-test ../../bin

cd ../../simple/simple-onnx
cargo build --release --target=wasm32-wasi
cp target/wasm32-wasi/release/simple-onnx.wasm ../../bin/simple-onnx.wasm

cd ../simple-test
cargo build --release
cp target/release/simple-test ../../bin
