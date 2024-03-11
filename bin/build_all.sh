#!/bin/bash

cd ../native
cargo build --release
cp target/release/onnx-native-test ../bin

cd ../wasm/wasi-nn
rustup target add wasm32-wasi
cargo build --release --target=wasm32-wasi
cp target/wasm32-wasi/release/wasi-nn-onnx-test.wasm ../../bin

cd ../wasmtime-test
cargo build --release
cp target/release/wasmtime-test ../../bin
