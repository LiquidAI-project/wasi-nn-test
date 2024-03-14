extern crate lazy_static;

use local_names::lazy_static::lazy_static;
use std::collections::HashMap;


const MODEL_FOLDER: &str = "models/";
const MODELS: [&str; 2] = [
    "mobilenetv2-10.onnx",
    "mobilenetv2-12.onnx"
];

const IMAGE_FOLDER: &str = "images/";
const IMAGES : [&str; 4] = [
    "landrover.jpg",
    "husky.jpg",
    "golden-retriever.jpg",
    "bigmac.png"
];


lazy_static! {
    static ref MODEL_MAP: HashMap<String, i32> = {
        let mut model_map = HashMap::new();
        for (index, &name) in MODELS.iter().enumerate() {
            model_map.insert(MODEL_FOLDER.to_owned() + name, (index + 1) as i32);
        }
        model_map
    };
}

lazy_static! {
    static ref IMAGE_MAP: HashMap<String, i32> = {
        let mut model_map = HashMap::new();
        for (index, &name) in IMAGES.iter().enumerate() {
            model_map.insert(IMAGE_FOLDER.to_owned() + name, (index + 1) as i32);
        }
        model_map
    };
}



fn get_index(map: &HashMap<String, i32>, name: &str) -> Option<i32> {
    match map.get(name) {
        Some(index) => Some(*index),
        None => None
    }
}

fn get_name(map: &HashMap<String, i32>, index: i32) -> Option<String> {
    for (name, &idx) in map.iter() {
        if idx == index {
            return Some(name.clone());
        }
    }
    None
}


pub fn get_model_index(name: &str) -> Option<i32> {
    get_index(&MODEL_MAP, name)
}

pub fn get_model_name(index: i32) -> Option<String> {
    get_name(&MODEL_MAP, index)
}

pub fn get_image_index(name: &str) -> Option<i32> {
    get_index(&IMAGE_MAP, name)
}

pub fn get_image_name(index: i32) -> Option<String> {
    get_name(&IMAGE_MAP, index)
}
