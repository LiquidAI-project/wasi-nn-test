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

For Wasmtime with ONNX runtime test:

```bash
cd bin
./wasmtime-test models/mobilenetv2-10.onnx images/landrover.jpg 100
```

Both models from `models` folder can be used in the tests and all three images from `images` folder. The last command line argument is the number of iterations done for multi-inference test.

## Things to check

- Consider modifying the preprocessing of the images in wasi-nn implementation. Currently based on wasi-nn-onnx implementation and is handled differently compared to the native implementation.
- Check the Wasmtime ONNX implementation at `wasm/wasmtime-onnx`. At the moment it is largely based on three-year old implementation from [https://github.com/deislabs/wasi-nn-onnx](https://github.com/deislabs/wasi-nn-onnx).
    - Could the Wasmtime ONNX implementation be based on [https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn)? As in add ONNX as supported runtime to that implementation.
        - Actually there is already a pull request intending to do just that: [https://github.com/bytecodealliance/wasmtime/pull/7691](https://github.com/bytecodealliance/wasmtime/pull/7691)
- Check how to switch from the `witx` generation to the more recent `wit` in Wasmtime ONNX.
- To allow utilization of the execution context for multiple inferences, some changes were made to the Wasmtime ONNX implementation:
    - Adding new input with the `set_input` method overwrites any existing input data. This likely is not the intended way it should be implemented.
    - Originally, running the `compute` method, when there was existing output caused an error. At the moment any existing output is just overwritten. Perhaps reading the output by calling the `get_output` method should also clear the output data from the context?
- Modify the native and wasi-nn implementations to have similar outputs.
