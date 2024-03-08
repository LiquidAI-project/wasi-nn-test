extern crate image2tensor;
extern crate wasi_nn;

use std::{cmp::Ordering, ops::RangeFrom};

use image2tensor::{ColorOrder, TensorType};
use wasi_nn::{Error, ExecutionTarget, Graph, GraphBuilder, GraphEncoding, GraphExecutionContext};


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


fn load_model(filename: &str) -> Result<Graph, Error> {
    GraphBuilder::new(
        GraphEncoding::Onnx,
        ExecutionTarget::CPU
    ).build_from_files([filename])
}


fn load_image(path: &str, width: u32, height: u32, precision: TensorType, color_order: ColorOrder) -> Result<Vec<u8>, String> {
    image2tensor::convert_image_to_tensor_bytes(path, width, height, precision, color_order)
}


fn get_result(model: &Graph, image_name: &str) -> Result<(f32, i32), ErrorType> {
    const MODEL_IMAGE_WIDTH: u32 = 224;
    const MODEL_IMAGE_HEIGHT: u32 = 224;
    const MODEL_IMAGE_PRECISION: TensorType = TensorType::F32;
    const MODEL_IMAGE_COLOR_ORDER: ColorOrder = ColorOrder::RGB;
    const MODEL_INPUT_DIMENSIONS: [usize; 4] = [1, 3, MODEL_IMAGE_WIDTH as usize, MODEL_IMAGE_HEIGHT as usize];

    let mut context: GraphExecutionContext<'_> = match model.init_execution_context() {
        Ok(context) => context,
        Err(_) => return Err(ErrorType::SessionCreation),
    };
    println!("Execution context created.");

    let image = match load_image(image_name, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_PRECISION, MODEL_IMAGE_COLOR_ORDER) {
        Ok(image) => image,
        Err(_) => return Err(ErrorType::ImageLoad),
    };
    println!("Image loaded: {:?}", image_name);

    match context.set_input(0, wasi_nn::TensorType::F32, &MODEL_INPUT_DIMENSIONS, &image) {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::ModelRun),
    }

    match context.compute() {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::ModelRun),
    }

    const OUTPUT_BUFFER_CAPACITY: usize = 4000;  // arbitrary max size
    let mut output_buffer: Vec<f32> = vec![0.0; OUTPUT_BUFFER_CAPACITY];
    match context.get_output(0, &mut output_buffer) {
        Ok(_) => (),
        Err(_) => return Err(ErrorType::TensorExtract),
    }

    output_buffer
        .iter()
        .cloned()
        .zip(RangeFrom::<i32>{start: 2})  // add the indexes for the labels
        .max_by(|(score1, _), (score2, _)| score1.partial_cmp(score2).unwrap_or(Ordering::Equal))
        .map_or_else(|| Err(ErrorType::NoResult), Ok)
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

fn get_model_filename(model_index: &i32) -> Option<&str> {
    match model_index {
        1 => Some("models/mobilenetv2-10.onnx"),
        2 => Some("models/mobilenetv2-12.onnx"),
        _ => None,
    }
}

fn get_image_name(image_index: &i32) -> Option<&str> {
    match image_index {
        1 => Some("images/landrover.jpg"),
        2 => Some("images/husky.jpg"),
        3 => Some("images/golden-retriever.jpg"),
        _ => None,
    }
}

#[no_mangle]
pub fn run_inference(model_index: i32, image_index: i32) -> i32 {
    let model_filename = match get_model_filename(&model_index) {
        Some(filename) => filename,
        None => {
            println!("Error: Invalid model index");
            return get_error_code(ErrorType::ModelLoad);
        }
    };

    let image_name = match get_image_name(&image_index) {
        Some(filename) => filename,
        None => {
            println!("Error: Invalid image index");
            return get_error_code(ErrorType::ImageLoad);
        }
    };

    println!("Running inference with model: {} and image: {}", model_filename, image_name);

    let model = match load_model(model_filename) {
        Ok(graph) => graph,
        Err(error) => {
            println!("Error loading model: {:?}", error);
            return get_error_code(ErrorType::ModelLoad);
        }
    };
    println!("Model loaded: {:?}", model_filename);

    let result = get_result(&model, image_name);
    println!("Result: {:?}", result);

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
    run_inference(1, 1);
}
