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
use std::{env, path::Path, time::Instant};


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

fn get_model_index(model_name: &str) -> Option<i32> {
    match model_name {
        "models/mobilenetv2-10.onnx" => Some(1),
        "models/mobilenetv2-12.onnx" => Some(2),
        _ => None,
    }
}

fn get_image_index(image_name: &str) -> Option<i32> {
    match image_name {
        "images/landrover.jpg" => Some(1),
        "images/husky.jpg" => Some(2),
        "images/golden-retriever.jpg" => Some(3),
        "images/bigmac.png" => Some(4),
        _ => None,
    }
}


fn main() -> wasmtime::Result<()> {
    let args: Vec<String> = env::args().collect();
    let model_filename: &str = &args[1];
    let image_name: &str = &args[2];
    let model_index = get_model_index(model_filename).unwrap();
    let image_index = get_image_index(image_name).unwrap();
    let repeats: u32 = args[3].parse().unwrap();

    const WASM_MODULE_FILENAME: &str = "wasi-nn-onnx-test.wasm";

    const MODEL_DIR: &str = "models";
    const IMAGE_DIR: &str = "images";
    let shared_dirs: Vec<&str> = vec![MODEL_DIR, IMAGE_DIR];

    let start: Instant = Instant::now();

    let config = Config::default();
    let engine = Engine::new(&config)?;
    let mut linker = wasmtime::Linker::new(&engine);

    wasi_common::sync::add_to_linker(&mut linker, |host: &mut Ctx| &mut host.wasi)?;
    wasmtime_onnx::add_to_linker(&mut linker, |host| &mut host.wasi_nn)?;

    let mut store = Store::new(
        &engine,
        Ctx::new(&shared_dirs)?
    );

    let environment_set_time = start.elapsed();

    println!("Loading module from file: {}", WASM_MODULE_FILENAME);
    let module = Module::from_file(&engine, WASM_MODULE_FILENAME)?;
    // println!("Module loaded successfully");

    // Print the expected imports
    // for import in module.imports() {
    //     println!("Module expects import with module '{}' and name '{}'", import.module(), import.name());
    // }

    // print the exports from the module
    // for export in module.exports() {
    //     println!("Exported function: {}", export.name());
    // }

    // add the module to the linker
    linker.module(&mut store, "wasi-nn", &module)?;

    let module_load_time = start.elapsed() - environment_set_time;

    let inference_function = linker
        .get(&mut store, "wasi-nn", "run_inference").unwrap()
        .into_func().unwrap()
        .typed::<(i32, i32), (i32,)>(&mut store).unwrap();
    let function_load_time = start.elapsed() - environment_set_time - module_load_time;
    // println!("Calling inference function");
    let result = inference_function.call(&mut store, (model_index, image_index));
    let inference_time = start.elapsed() -environment_set_time - module_load_time - function_load_time;
    println!("Result: {:?}", result);

    println!("Environment set time: {:?}", environment_set_time);
    println!("Module load time: {:?}", module_load_time);
    println!("Function load time: {:?}", function_load_time);
    println!("Inference time: {:?}", inference_time);

    println!("");

    // test the repeated inference performance
    let start2: Instant = Instant::now();
    let inference_multiple_function = linker
        .get(&mut store, "wasi-nn", "run_multiple_inference").unwrap()
        .into_func().unwrap()
        .typed::<(i32, i32, u32), (f32,)>(&mut store).unwrap();
    let function2_load_time = start2.elapsed();
    let result2 = inference_multiple_function.call(&mut store, (model_index, image_index, repeats));
    println!("Result for multiple inference: {:?}", result2);
    let multiple_inference_time = start2.elapsed() - function2_load_time;
    println!("Multiple inference time: {:?}", multiple_inference_time);

    Ok(())
}
