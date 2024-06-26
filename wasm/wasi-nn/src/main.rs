extern crate image;
// extern crate image2tensor;
extern crate wasi_nn;
extern crate anyhow;
extern crate ndarray;
extern crate local_names;

use anyhow::Error;
use std::{cmp::Ordering, ops::RangeFrom, time::Instant};
use image::{imageops::FilterType, Pixel};
use ndarray::s;
// use image2tensor::{ColorOrder, TensorType};
use wasi_nn::{ExecutionTarget, Graph, GraphBuilder, GraphEncoding, GraphExecutionContext};
use local_names::{get_image_name, get_model_name};


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


fn load_model(filename: String) -> Result<Graph, wasi_nn::Error> {
    GraphBuilder::new(
        GraphEncoding::Onnx,
        ExecutionTarget::GPU
    ).build_from_files([filename])
}


fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let mut result: Vec<u8> = Vec::new();

    // TODO
    // simplify this to potentially a single map.
    for c in chunks {
        for u in c.iter() {
            result.push(*u);
        }
    }
    result
}

pub fn image_to_tensor(
    path: &str,
    height: u32,
    width: u32,
) -> Result<Vec<u8>, Error> {
    let image = image::imageops::resize(
        &image::open(path)?,
        width,
        height,
        ::FilterType::Triangle,
    );

    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    Ok(f32_vec_to_bytes(array.as_slice().unwrap().to_vec()))
}


fn load_image(path: String, width: u32, height: u32) -> Result<Vec<u8>, String> {
// fn load_image(path: &str, width: u32, height: u32, precision: TensorType, color_order: ColorOrder) -> Result<Vec<u8>, String> {
    // image2tensor::convert_image_to_tensor_bytes(path, width, height, precision, color_order)
    image_to_tensor(&path, width, height).map_err(|e| e.to_string())
}


fn get_execution_context(model: &Graph) -> Result<GraphExecutionContext<'_>, ErrorType> {
    match model.init_execution_context() {
        Ok(context) => Ok(context),
        Err(_) => Err(ErrorType::SessionCreation),
    }
}


fn get_result(
    // model: &Graph,
    context: &mut GraphExecutionContext<'_>,
    image_name: String,
    verbose: bool
) -> Result<(f32, i32), ErrorType> {
    const MODEL_IMAGE_WIDTH: u32 = 224;
    const MODEL_IMAGE_HEIGHT: u32 = 224;
    // const MODEL_IMAGE_PRECISION: TensorType = TensorType::F32;
    // const MODEL_IMAGE_COLOR_ORDER: ColorOrder = ColorOrder::RGB;
    const MODEL_INPUT_DIMENSIONS: [usize; 4] = [1, 3, MODEL_IMAGE_WIDTH as usize, MODEL_IMAGE_HEIGHT as usize];

    let result_start: Instant = Instant::now();

    let image = match load_image(image_name, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT) {
    // let image = match load_image(image_name, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_PRECISION, MODEL_IMAGE_COLOR_ORDER) {
        Ok(image) => image,
        Err(_) => return Err(ErrorType::ImageLoad),
    };
    let image_load_time = result_start.elapsed();

    match context.set_input(0, wasi_nn::TensorType::F32, &MODEL_INPUT_DIMENSIONS, &image) {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::ModelRun),
    }
    let input_set_time = result_start.elapsed() - image_load_time;

    match context.compute() {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::ModelRun),
    }
    let model_run_time = result_start.elapsed() - input_set_time - image_load_time;

    const OUTPUT_BUFFER_CAPACITY: usize = 4000;  // arbitrary max size
    let mut output_buffer: Vec<f32> = vec![0.0; OUTPUT_BUFFER_CAPACITY];
    match context.get_output(0, &mut output_buffer) {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::TensorExtract),
    }
    let tensor_extract_time = result_start.elapsed() - model_run_time - input_set_time - image_load_time;

    let result = output_buffer
        .iter()
        .cloned()
        .zip(RangeFrom::<i32>{start: 1})  // add the indexes for the labels
        .max_by(|(score1, _), (score2, _)| score1.partial_cmp(score2).unwrap_or(Ordering::Equal))
        .map_or_else(|| Err(ErrorType::NoResult), Ok);
    let result_calculation_time = result_start.elapsed() - tensor_extract_time - model_run_time - input_set_time - image_load_time;

    if verbose {
        println!("Loading the image took {:?}", image_load_time);
        println!("Running the inference took {:?}", input_set_time + model_run_time);
        println!("Extracting the result took {:?}", tensor_extract_time + result_calculation_time);
    }

    result
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

    // println!("Running inference with model: {} and image: {}", model_filename, image_name);
    let start: Instant = Instant::now();

    let model = match load_model(model_filename) {
        Ok(graph) => graph,
        Err(error) => {
            println!("Error loading model: {:?}", error);
            return get_error_code(ErrorType::ModelLoad);
        }
    };
    let model_load_time = start.elapsed();
    println!("Loading the model took {:?}", model_load_time);

    let mut context = match get_execution_context(&model) {
        Ok(context) => context,
        Err(error) => {
            println!("Error creating context: {:?}", error);
            return get_error_code(error);
        }
    };
    let context_creation_time = start.elapsed() - model_load_time;
    println!("Execution context creation took {:?}", context_creation_time);

    let result = get_result(&mut context, image_name.clone(), true);
    let result_calculation_time = start.elapsed() - model_load_time - context_creation_time;

    for _ in 0..repeats {
        let _ = get_result(&mut context, image_name.clone(), false);
    }
    let repeat_time = start.elapsed() - model_load_time - context_creation_time - result_calculation_time;
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
