# Test repository for wasi-nn

This repository contains a test programs for running ML inference with Wasm module using wasi-nn API such that it utilizes the host machine capabilities for getting the result. As comparison, there is also a program for running the inference natively on the host machine as well as a program which runs the entire inference within the Wasm module.

Contents of the repository:

- [native](native/) folder
    - Contains a Rust program that runs inference natively using an ONNX model.
- [wasm/local-names](wasm/local-names/) folder
    - Contains a helper Rust program that transforms a filename for a model or an image to an index, or vice versa.
- [wasm/wasi-nn](wasm/wasi-nn/) folder
    - Contains a Rust program that runs inference with ONNX model using the wasi-nn API.
    - Should be compiled to the wasm32-wasi target.
- [wasm/wasmtime-test](wasm/wasmtime-test/) folder
    - Contains a Rust program that loads in a Wasm module with [Wasmtime](https://wasmtime.dev/) and runs inference using the module.
    - Provides the WASI and wasi-nn host interfaces to the Wasm module. WASI interface for accessing the file system, and wasi-nn interface for utilizing the host ML capabilities.
    - Utilizes code from the wasmtime submodule which provides ONNX support for Wasmtime and wasi-nn. (And is not available yet in the release at [https://crates.io/crates/wasmtime-wasi-nn](https://crates.io/crates/wasmtime-wasi-nn)).
- [wasm/simple-onnx](wasm/simple-onnx/) folder
    - Contains a Rust programs that runs inference with ONNX model using [tract](https://github.com/sonos/tract) runtime without utilizing the host capabilities.
    - Should be compiled to the wasm32-wasi target. The program uses tract instead of ort (like other inference code) because ort cannot be compiled to the wasi target.
- [wasmtime-repo](wasmtime-repo/) Git submodule for [wasmtime](https://github.com/bytecodealliance/wasmtime)
- [bin](bin/) folder
    - Contains a build script for all test programs.
    - Contains test ML models (different mobilenet version in ONNX format).
    - Contains test images for the inference.
- [legacy](legacy/) folder
    - Contains earlier versions of the test program that are not in use any more.
    - [wasmtime-onnx](legacy/wasmtime-onnx/) contains earlier ONNX support for Wasmtime and wasi-nn which was replaced by [wasmtime-wasi-nn](https://github.com/bytecodealliance/wasmtime/tree/main/crates/wasi-nn). This also utilized the definitions from the `spec` folder.
    - [simple-onnx](legacy/simple-test/) contains a simpler version of wasmtime-test which only provided the WASI interfaces and not the wasi-nn interfaces.

Currently, only CPU is supported, but various GPUs should be usable with minor modifications with the used framework.

## Build instructions

Perquisites (possibly incomplete list):

- [Rust](https://www.rust-lang.org/) for compiling the programs.
    - Tested with the latest stable channel version (1.88.0).
- [Bash](https://www.gnu.org/software/bash/) for running the build script.
    - On Windows the [Git Bash](https://gitforwindows.org/) can be used.

Testes on WSL 2 (with Ubuntu 24.04) and Windows 11 operating systems.

```bash
# clone the repository with the recursive flag to get all submodule data
git clone --recursive git@github.com:LiquidAI-project/wasi-nn-test.git
cd wasi-nn-test

# run the build script in the bin folder
cd bin
./build_all.sh
```

## Run instructions

For native ONNX runtime test:

```bash
# in the bin folder
./onnx-native-test models/mobilenetv2-10.onnx images/husky.jpg 100
```

For Wasmtime with ONNX runtime with wasi-nn test:

```bash
# in the bin folder
./wasmtime-test wasi-nn-onnx-test.wasm models/mobilenetv2-10.onnx images/husky.jpg 100
```

For Wasmtime with Tract runtime without wasi-nn test:

```bash
# in the bin folder
./wasmtime-test simple-onnx.wasm models/mobilenetv2-10.onnx images/husky.jpg 10
```

Both models from `models` folder can be used in the tests and all three images from `images` folder. The last command line argument is the number of iterations done for multi-inference test.

## Things to check

- Consider modifying the preprocessing of the images in wasi-nn implementation. Currently based on wasi-nn-onnx implementation and is handled differently compared to the native implementation.
- Add a way to provide the configuration parameters for the execution context in order to also support GPU usage for the inference.
