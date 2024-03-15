extern crate lazy_static;
extern crate glob;

use local_names::glob::glob;
use local_names::lazy_static::lazy_static;
use std::collections::HashMap;
use std::path::PathBuf;


const MODEL_FOLDER: &str = "models/";
const IMAGE_FOLDER: &str = "images/";


fn get_hashmap(folder: &str, pattern: &str) -> HashMap<String, i32> {
    let mut output_map = HashMap::new();
    let file_list: Vec<PathBuf> = match glob(&(folder.to_owned() + pattern)) {
        Ok(paths) => paths.filter_map(Result::ok).collect(),
        Err(_) => Vec::new(),
    };
    for (index, name) in file_list.iter().enumerate() {
        if let Some(name_str) = name.to_str() {
            // Replace backslashes with forward slashes for Windows support
            output_map.insert(name_str.to_string().replace("\\", "/"), (index + 1) as i32);
        }
    }
    output_map
}

lazy_static! {
    #[derive(Debug)]
    static ref MODEL_MAP: HashMap<String, i32> = get_hashmap(MODEL_FOLDER, "*.onnx");
}

lazy_static! {
    static ref IMAGE_MAP: HashMap<String, i32> = get_hashmap(IMAGE_FOLDER, "*.*");
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
