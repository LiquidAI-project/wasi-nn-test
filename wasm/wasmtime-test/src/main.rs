extern crate wasmtime;
extern crate wasmtime_wasi;
extern crate wasi_common;
extern crate wasmtime_onnx;
extern crate anyhow;
extern crate cap_std;

use anyhow::{Ok, Result};
use wasmtime::{Config, Engine, Module, Store};
use wasi_common::{sync::Dir, sync::WasiCtxBuilder, WasiCtx};
use wasmtime_onnx::WasiNnOnnxCtx;
use std::path::Path;


/// The host state for running wasi-nn tests.
struct Ctx {
    wasi: WasiCtx,
    wasi_nn: WasiNnOnnxCtx,
}
impl Ctx {
    fn new(directories: &Vec<&str>) -> Result<Self> {
        let preopen_dirs = directories
            .iter()
            .map(|dir| {
                Dir::open_ambient_dir(Path::new(dir), cap_std::ambient_authority())
            }.unwrap());

        let mut binding = WasiCtxBuilder::new();
        let builder = binding.inherit_stdio();
        for (preopen_dir, path) in preopen_dirs.zip(directories) {
            builder.preopened_dir(preopen_dir, path)?;
        }

        let wasi = builder.build();
        let wasi_nn = WasiNnOnnxCtx::default();

        Ok(Self { wasi, wasi_nn })
    }
}


fn main() -> wasmtime::Result<()> {
    const WASM_MODULE_FILENAME: &str = "wasi-nn-onnx-test.wasm";

    const MODEL_DIR: &str = "models";
    const IMAGE_DIR: &str = "images";
    let shared_dirs: Vec<&str> = vec![MODEL_DIR, IMAGE_DIR];

    let config = Config::default();
    let engine = Engine::new(&config)?;
    let mut linker = wasmtime::Linker::new(&engine);

    wasi_common::sync::add_to_linker(&mut linker, |host: &mut Ctx| &mut host.wasi)?;
    wasmtime_onnx::add_to_linker(&mut linker, |host| &mut host.wasi_nn)?;

    let mut store = Store::new(
        &engine,
        Ctx::new(&shared_dirs)?
    );

    println!("Loading module from file: {}", WASM_MODULE_FILENAME);
    let module = Module::from_file(&engine, WASM_MODULE_FILENAME)?;
    println!("Module loaded successfully");

    // Print the expected imports
    for import in module.imports() {
        println!("Module expects import with module '{}' and name '{}'", import.module(), import.name());
    }

    // print the exports from the module
    for export in module.exports() {
        println!("Exported function: {}", export.name());
    }

    // add the module to the linker
    let linker_update = linker.module(&mut store, "wasi-nn", &module);
    println!("Linker updated: {:?}", linker_update.err());

    let inference_function = linker
        .get(&mut store, "wasi-nn", "run_inference").unwrap()
        .into_func().unwrap()
        .typed::<(i32, i32), (i32,)>(&mut store).unwrap();
    println!("Calling inference function");
    let result = inference_function.call(&mut store, (1, 1));
    println!("Result: {:?}", result);
    Ok(())
}
