extern crate local_names;
extern crate tract_onnx;

use std::time::Instant;
use tract_onnx::{
    self as tonnx,
    prelude::{self as tp, tvec, Framework, InferenceModelExt, Tensor},
    tract_core::ops::TypedOp,
    tract_hir::prelude::{SimplePlan, TypedFact, Graph}
};
use local_names::{get_image_name, get_model_name};

type RunnableModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;


#[derive(Debug)]
pub enum ErrorType {
    SessionCreation,
    Optimization,
    Threads,
    ModelLoad,
    ImageLoad,
    ImageConversion,
    ModelRun,
    TensorExtract,
    NoResult,
    MissingImageName,
}


fn get_error_code(error: ErrorType) -> i32 {
    match error {
        ErrorType::SessionCreation => -1,
        ErrorType::Optimization => -2,
        ErrorType::Threads => -3,
        ErrorType::ModelLoad => -4,
        ErrorType::ImageLoad => -5,
        ErrorType::ImageConversion => -6,
        ErrorType::ModelRun => -7,
        ErrorType::TensorExtract => -8,
        ErrorType::NoResult => -9,
        ErrorType::MissingImageName => -10,
    }
}


// Adapted from:
// https://github.com/sonos/tract/blob/fd690600d1993ca4b90e2e73db4c5dccbecf5ded/examples/onnx-mobilenet-v2/src/main.rs
// and based further on:
// https://github.com/LiquidAI-project/wasmiot-modules/blob/main/modules/wasi_mobilenet_inference_onnx/src/inference.rs
fn get_result(runnable_model: &RunnableModel, image_name: String, verbose: bool) -> Result<(f32, i32), ErrorType> {
    let result_start = Instant::now();

    let image = match image::open(&image_name) {
        Ok(image) => image.to_rgb8(),
        Err(e) => {
            println!("{:?}", e);
            return Err(ErrorType::ImageLoad);
        }
    };
    let resized =
        image::imageops::resize(&image, 224, 224, image::imageops::FilterType::Triangle);
    let image: Tensor = tp::tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224),  |(_, c, y, x)| {
        let mean = [0.485, 0.456, 0.406][c];
        let std = [0.229, 0.224, 0.225][c];
        (resized[(x as _, y as _)][c] as f32 / 255.0 - mean) / std
    }).into();

    let image_load_time = result_start.elapsed();

    let result = match runnable_model.run(tvec!(image.into())) {
        Ok(result) => result,
        Err(error) => {
            eprintln!("{:?}", error);
            return Err(ErrorType::ModelRun);
        }
    };
    let model_run_time = result_start.elapsed() - image_load_time;

    let final_result = match result[0].to_array_view::<f32>() {
        Ok(array_result) =>
            match array_result.iter()
                .cloned()
                .zip(1..)
                .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()) {
                    Some(label_result) => label_result,
                    None => return Err(ErrorType::NoResult),
                },
        Err(_) => {
            return Err(ErrorType::TensorExtract);
        }
    };
    let result_calculation_time = result_start.elapsed() - model_run_time - image_load_time;

    if verbose {
        println!("Loading the image took {:?}", image_load_time);
        println!("Running the inference took {:?}", model_run_time);
        println!("Extracting the result took {:?}", result_calculation_time);
    }

    Ok(final_result)

}

#[no_mangle]
pub fn run_inference(model_index: i32, image_index: i32, repeats: u32) -> i32 {
    let model_filename = match get_model_name(model_index) {
        Some(filename) => filename,
        None => {
            println!("Error: Invalid model index");
            return get_error_code(ErrorType::ModelLoad);
        }
    };

    let image_name = match get_image_name(image_index) {
        Some(filename) => filename,
        None => {
            println!("Error: Invalid image index");
            return get_error_code(ErrorType::ImageLoad);
        }
    };

    let start: Instant = Instant::now();

    let model_input = match tonnx::onnx().model_for_path(&model_filename) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("{:?}", e);
            return get_error_code(ErrorType::ModelLoad);
        }
    };
    let model_optimized = match model_input.into_optimized() {
        Ok(model) => model,
        Err(_) => {
            return get_error_code(ErrorType::Optimization);
        }
    };
    let model_runnable: RunnableModel = match model_optimized.into_runnable() {
        Ok(model) => model,
        Err(_) => {
            return get_error_code(ErrorType::SessionCreation);
        }
    };
    let model_load_time = start.elapsed();
    println!("Loading the model took {:?}", model_load_time);

    let result = get_result(&model_runnable, image_name.clone(), true);
    let result_calculation_time = start.elapsed() - model_load_time;

    for _ in 0..repeats {
        let _ = get_result(&model_runnable, image_name.clone(), false);
    }
    let repeat_time = start.elapsed() - model_load_time - result_calculation_time;
    println!("\nRunning the model {} times took {:?}\n", repeats, repeat_time);

    match result {
        Ok((score, class)) => {
            println!("{}: {} (score: {})", image_name, class, score);
            class
        },
        Err(error) => {
            println!("Error: {:?}", error);
            get_error_code(error)
        }
    }
}



fn main() {
    run_inference(1, 1, 10);
}
