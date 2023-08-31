cargo_component_bindings::generate!();

use image::io::Reader;
use image::DynamicImage;
use ndarray::{Array, Dim};
use std::fs;

const IMG_PATH: &str = "fixture/images/0.jpg";

/// Generate the traits and types from the `wasi-nn` WIT specification.
mod wit {
    wit_bindgen::generate!({path: "../../spec/wit/wasi-nn.wit"});
}

use crate::wit::wasi::nn::inference::{compute, get_output};
use wit::wasi::nn::graph::{load, ExecutionTarget, GraphBuilder, GraphEncoding};
use wit::wasi::nn::inference::{init_execution_context, set_input};
use wit::wasi::nn::tensor::{Tensor, TensorData, TensorDimensions, TensorType};

pub fn main() {
    let model: GraphBuilder = fs::read("fixture/models/sqeezenet1.0-12.onnx").unwrap();
    println!("Read ONNX model, size in bytes: {}", model.len());

    let graph = load(&[model], GraphEncoding::Onnx, ExecutionTarget::Cpu).unwrap();
    println!("Loaded graph into wasi-nn with ID: {}", graph);

    let exec_context = init_execution_context(graph).unwrap();
    println!("Created wasi-nn execution context.");

    let dimensions: TensorDimensions = vec![1, 3, 224, 224];
    let data: TensorData = image_to_tensor(IMG_PATH.to_string(), 224, 224);
    let tensor = Tensor {
        dimensions,
        tensor_type: TensorType::Fp32,
        data,
    };
    set_input(exec_context, 0, &tensor).unwrap();
    println!("Set input tensor.");

    compute(exec_context).unwrap();
    println!("Executed graph inference");

    println!("Getting output 0");
    let output_data = get_output(exec_context, 0).unwrap();
    println!("Retrieved output data with length: {}", output_data.len());
    let output_f32 = bytes_to_f32_vec(output_data);

    let output_shape = [1, 1000, 1, 1];
    let output_tensor = Array::from_shape_vec(output_shape, output_f32).unwrap();

    let mut sorted = output_tensor
        .axis_iter(ndarray::Axis(1))
        .enumerate()
        .into_iter()
        .map(|(i, v)| (i, v[Dim([0, 0, 0])]))
        .collect::<Vec<(_, _)>>();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for i in 0..5 {
        println!("class={}; probability={}", sorted[i].0, sorted[i].1);
    }
    println!("Done!!")
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
fn image_to_tensor(path: String, height: u32, width: u32) -> Vec<u8> {
    let pixels = Reader::open(path).unwrap().decode().unwrap();
    let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
    let bgr_img = dyn_img.to_rgb8();
    // Get an array of the pixel values
    let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
    // Create an array to hold the f32 value of those pixels
    let bytes_required = raw_u8_arr.len() * 4;
    let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

    for i in 0..raw_u8_arr.len() {
        // Read the number as a f32 and break it into u8 bytes
        let u8_f32: f32 = raw_u8_arr[i] as f32;
        let u8_bytes = u8_f32.to_ne_bytes();

        for j in 0..4 {
            u8_f32_arr[(i * 4) + j] = u8_bytes[j];
        }
    }
    return u8_f32_arr;
}
