# Native ONNX inference

Inference using ONNX models implemented with Rust library [ort](https://github.com/pykeio/ort) that implements the Rust bindings for [ONNX Runtime](https://github.com/microsoft/onnxruntime).

## Installing Rust

On Ubuntu 22.04:

```bash
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get install -y curl build-essential

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rustup.sh && sh rustup.sh -y && rm rustup.sh
echo "$HOME/.cargo/env" >> .bashrc
. "$HOME/.cargo/env"
```

## Compilation

- Release build: `cargo build --profile release`
- Debug build: `cargo build --profile dev`

Cross compilation with release build from Ubuntu 22.04 on amd64 architecture (only building has been tested):

- To `aarch64-unknown-linux-gnu`:

    ```bash
    sudo apt-get install -y gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
    rustup target add aarch64-unknown-linux-gnu

    cargo build --target=aarch64-unknown-linux-gnu --profile release
    ```

- To `armv7-unknown-linux-gnueabihf` (WIP):

    ```bash
    sudo apt-get install -y gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf
    rustup target add armv7-unknown-linux-gnueabihf

    # compile the onnxruntime binaries (will take a while and require several GBs of space)
    sudo apt-get install -y git cmake python3 python3-pip
    git clone --recursive https://github.com/Microsoft/onnxruntime.git onnxruntime
    cd onnxruntime
    # the build command needs to be adjusted (WIP)
    ./build.sh --config Release --parallel --compile_no_warning_as_error --skip_submodule_sync --arm
    cd ..

    ORT_LIB_LOCATION=onnxruntime/build/Linux cargo build --target=armv7-unknown-linux-gnueabihf --profile release
    ```
