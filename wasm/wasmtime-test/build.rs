use std::process::Command;

macro_rules! p {
    ($($tokens: tt)*) => {
        println!("cargo:info={}", format!($($tokens)*))
    }
}

fn main() {
    // A really ad-hoc solution to fix the issue with the onnxruntime.rs file from the submodule
    // - better solution would be have the original code fixed
    // - or at least have a fork of the submodule with the fix as the source for this crate

    let output1 = Command::new("cp")
        .arg("wasi-nn/Cargo.toml")
        .arg("../../wasmtime-repo/crates/wasi-nn/Cargo.toml")
        .output()
        .expect("Failed to execute command");

    let output2 = Command::new("cp")
        .arg("wasi-nn/onnxruntime.rs")
        .arg("../../wasmtime-repo/crates/wasi-nn/src/backend/onnxruntime.rs")
        .output()
        .expect("Failed to execute command");

    // these outputs may or may not be visible
    p!("sed 1: {:?}", output1);
    p!("sed 2: {:?}", output2);
}
