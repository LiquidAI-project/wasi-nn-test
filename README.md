# Test repository for wasi-nn

## Build instructions

```bash
cd bin
./build_all.sh
```

## Run instructions

For native ONNX runtime test:

```bash
cd bin
./onnx-native-test models/mobilenetv2-10.onnx images/landrover.jpg 100
```

For Wasmtime with ONNX runtime with wasi-nn test:

```bash
cd bin
./wasmtime-test wasi-nn-onnx-test.wasm models/mobilenetv2-10.onnx images/husky.jpg 100
```

For Wasmtime with Tract runtime without wasi-nn test:

```bash
cd bin
./wasmtime-test simple-onnx.wasm models/mobilenetv2-10.onnx images/husky.jpg 10
```

Both models from `models` folder can be used in the tests and all three images from `images` folder. The last command line argument is the number of iterations done for multi-inference test.

## Things to check

- Consider modifying the preprocessing of the images in wasi-nn implementation. Currently based on wasi-nn-onnx implementation and is handled differently compared to the native implementation.
- Add a way to provide the configuration parameters for the execution context in order to also support GPU usage for the inference.
