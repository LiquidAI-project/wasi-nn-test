#!/bin/bash

echo "========================"
echo "Running native ONNX test"
echo "========================"
./onnx-native-test models/mobilenetv2-10.onnx images/husky.jpg 100

echo ""
echo "========================="
echo "Running wasi-nn ONNX test"
echo "========================="
./wasmtime-test wasi-nn-onnx-test.wasm models/mobilenetv2-10.onnx images/husky.jpg 100

echo ""
echo "============================="
echo "Running simple Wasm ONNX test"
echo "============================="
./wasmtime-test simple-onnx.wasm models/mobilenetv2-10.onnx images/husky.jpg 10
