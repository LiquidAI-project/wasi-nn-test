#!/bin/bash

cd ../native
cargo clean
cd ../wasm/wasi-nn
cargo clean
cd ../simple-onnx
cargo clean
cd ../wasmtime-test
cargo clean
cd ../../bin
