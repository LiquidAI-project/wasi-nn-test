extern crate image;
extern crate ort;
extern crate ndarray;

use std::{env, cmp::Ordering, convert::TryInto, ops::RangeFrom, time::{Duration, Instant}};
use image::{imageops::FilterType, ImageBuffer, ImageError, Rgb};
use ndarray::{Array4, OwnedRepr, prelude::{ArrayBase, Dim}};
use ort::{GraphOptimizationLevel, Session, Tensor, Value};

type RawImage = ImageBuffer<Rgb<u8>, Vec<u8>>;
type NormalizedImage = ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>>;

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


fn load_model(filename: &str) -> Result<Session, ErrorType> {
    Session::builder()
        .map_err(|error| {
            eprintln!("Error creating session: {:?}", error);
            ErrorType::SessionCreation
        })?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|error| {
            eprintln!("Error setting optimization level: {:?}", error);
            ErrorType::Optimization
        })?
        // .with_inter_threads(12)
        // .map_err(|error| {
        //     eprintln!("Error setting inter threads: {:?}", error);
        //     ErrorType::Threads
        // })?
        // .with_intra_threads(12)
        // .map_err(|error| {
        //     eprintln!("Error setting intra threads: {:?}", error);
        //     ErrorType::Threads
        // })?
        .with_model_from_file(filename)
        .map_err(|error| {
            eprintln!("Error loading model: {:?}", error);
            ErrorType::ModelLoad
        })
}

fn load_image(path: &str, nwidth: u32, nheight: u32, filter: FilterType) -> Result<Value, ErrorType> {
    let image: Result<RawImage, ImageError> = image::open(path)
        .map(|image| image.resize_exact(nwidth, nheight, filter).to_rgb8());

    match image {
        Ok(image) => {
            let normalized_image: NormalizedImage = Array4::from_shape_fn(
                (1, 3, nwidth as usize, nheight as usize),
                |(_, color, y, x)| {
                    // color normalization magic from
                    // https://github.com/sonos/tract/tree/fd690600d1993ca4b90e2e73db4c5dccbecf5ded/examples/onnx-mobilenet-v2
                    let mean = [0.485, 0.456, 0.406][color];
                    let std = [0.229, 0.224, 0.225][color];
                    (image[(x as u32, y as u32)][color] as f32 / 255.0 - mean) / std
                }
            );

            TryInto::<Value>::try_into(normalized_image)
                .map_err(|error| {
                    eprintln!("Error converting image to input value: {:?}", error);
                    ErrorType::ImageConversion
                })
        },
        Err(error) => {
            eprintln!("Error loading image: {:?}", error);
            Err(ErrorType::ImageLoad)
        }
    }
}

fn run_model<'model>(model: &'model Session, image: Value) -> Result<ort::SessionOutputs<'model>, ErrorType> {
    // let cloned_image = image.clone();
    model.run([image])
        .map_err(|error| {
            eprintln!("Error running model: {:?}", error);
            ErrorType::ModelRun
        })
}

fn get_result(model: &Session, image_name: &str, verbose: bool) -> Result<(f32, i32), ErrorType> {
    const MODEL_IMAGE_WIDTH: u32 = 224;
    const MODEL_IMAGE_HEIGHT: u32 = 224;
    const MODEL_IMAGE_FILTER_TYPE: FilterType = FilterType::Triangle;

    // load the image and run the model
    let result_start: Instant = Instant::now();
    let image_result = load_image(image_name, MODEL_IMAGE_WIDTH, MODEL_IMAGE_HEIGHT, MODEL_IMAGE_FILTER_TYPE);
    let image_load_duration: Duration = result_start.elapsed();
    let model_output = image_result.and_then(|image| run_model(&model, image))?;
    let model_run_duration: Duration = result_start.elapsed() - image_load_duration;

    // extract the results
    let model_results = model_output
        .iter()
        .map(|(_, output_value)| output_value.extract_tensor::<f32>().map_err(|error| {
            eprintln!("Error extracting tensor: {:?}", error);
            ErrorType::TensorExtract
        }))
        .collect::<Result<Vec<Tensor<f32>>, ErrorType>>()?;
    let temp_result = model_results
        .first()
        .map_or_else(|| Err(ErrorType::NoResult), Ok);
    let result = match temp_result {
        Ok(value) => value,
        Err(error) => return Err(error),
    };

    // find the highest score and the corresponding label
    let final_result = result
        .view()
        .iter()
        .cloned()
        .zip(RangeFrom::<i32>{start: 1})  // add the indexes for the labels
        .max_by(|(score1, _), (score2, _)| score1.partial_cmp(score2).unwrap_or(Ordering::Equal))
        .map_or_else(|| Err(ErrorType::NoResult), Ok);
    let final_duration: Duration = result_start.elapsed() - image_load_duration - model_run_duration;

    if verbose {
        println!("Loading the image took {:?}", image_load_duration);
        println!("Running the inference took {:?}", model_run_duration);
        println!("Extracting the result took {:?}", final_duration);
    }

    final_result
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

pub fn main() -> Result<(), i32> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        println!("Usage: {} <model> <image> <number of repeats2>", args[0]);
        return Err(-10);
    }

    let model_filename: &str = &args[1];
    let image_name: &str = &args[2];
    let repeats: u32 = args[3].parse().unwrap();

    // initialize the environment
    let start: Instant = Instant::now();
    let used_execution_providers = [
        // ort::CUDAExecutionProvider::default().build(),
        // ort::TensorRTExecutionProvider::default().build(),
        // ort::OpenVINOExecutionProvider::default().build(),
        ort::CPUExecutionProvider::default().build()
    ];
    let _ = match ort::init()
        .with_execution_providers(used_execution_providers)
        .commit() {
        Ok(_) => (),
        Err(error) => {
            eprintln!("Error initializing ONNXRuntime: {:?}", error);
            return Err(get_error_code(ErrorType::SessionCreation));
        }
    };
    let environment_duration: Duration = start.elapsed();
    println!("Initializing the environment took {:?}", environment_duration);

    let model_result = load_model(model_filename);
    let duration1 = start.elapsed() - environment_duration;
    println!("Loading the model took {:?}", duration1);

    let model = match model_result {
        Ok(session) => session,
        Err(error) => return Err(get_error_code(error)),
    };

    let result = get_result(&model, image_name, true);
    let duration2: Duration = start.elapsed() - environment_duration - duration1;

    for _ in 0..repeats {
        let _ = get_result(&model, image_name, false);
    }
    let duration3: Duration = start.elapsed() - environment_duration - duration1 - duration2;

    println!("\nRunning the model {} times took {:?}\n", repeats, duration3);

    match result {
        Ok((score, class)) => {
            println!("{}: {} (score: {})", image_name, class, score);
            Ok(())
        },
        Err(error) => {
            println!("Error: {:?}", error);
            Err(get_error_code(error))
        }
    }
}
