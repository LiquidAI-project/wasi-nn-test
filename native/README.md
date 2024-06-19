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
