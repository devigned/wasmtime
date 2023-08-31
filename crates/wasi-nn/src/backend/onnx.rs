//! Implements a `wasi-nn` [`BackendInner`] using ONNX.

use super::{BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner};
use crate::backend::read;
use crate::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use crate::{ExecutionContext, Graph};
use lazy_static::lazy_static;
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::Array;
use onnxruntime::session::Session;
use onnxruntime::tensor::OrtOwnedTensor;
use onnxruntime::{GraphOptimizationLevel, OrtError};
use std::path::Path;
use std::sync::{Arc, Mutex};
extern crate lazy_static;

#[derive(Default)]
pub struct OnnxBackend(Option<Environment>);
unsafe impl Send for OnnxBackend {}
unsafe impl Sync for OnnxBackend {}

lazy_static! {
    static ref ONNX_ENV: Environment = Environment::builder().with_name("env").build().unwrap();
}

impl BackendInner for OnnxBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Onnx
    }

    fn load(&mut self, builders: &[&[u8]], target: ExecutionTarget) -> Result<Graph, BackendError> {
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()).into());
        }

        let session = ONNX_ENV
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Basic)
            .unwrap()
            .with_model_from_memory(builders[0])
            .unwrap();

        let box_: Box<dyn BackendGraph> =
            Box::new(ONNXGraph(Arc::new(Mutex::new(session)), target));
        Ok(box_.into())
    }

    fn as_dir_loadable(&mut self) -> Option<&mut dyn BackendFromDir> {
        Some(self)
    }
}

impl BackendFromDir for OnnxBackend {
    fn load_from_dir(
        &mut self,
        path: &Path,
        target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        let model = read(&path.join("model.onnx"))?;
        self.load(&[&model], target)
    }
}

struct ONNXGraph(Arc<Mutex<Session<'static>>>, ExecutionTarget);

unsafe impl Send for ONNXGraph {}
unsafe impl Sync for ONNXGraph {}

impl BackendGraph for ONNXGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let inputs = self
            .0
            .lock()
            .unwrap()
            .inputs
            .iter()
            .map(|_| None)
            .collect::<Vec<_>>();
        let outputs = self
            .0
            .lock()
            .unwrap()
            .outputs
            .iter()
            .map(|_| None)
            .collect::<Vec<_>>();
        let box_: Box<dyn BackendExecutionContext> = Box::new(ONNXExecutionContext {
            session: self.0.clone(),
            inputs,
            outputs,
        });
        Ok(box_.into())
    }
}

struct ONNXExecutionContext<'a> {
    session: Arc<Mutex<Session<'a>>>,
    inputs: Vec<Option<Tensor>>,
    outputs: Vec<Option<Vec<u8>>>,
}

unsafe impl Send for ONNXExecutionContext<'_> {}
unsafe impl Sync for ONNXExecutionContext<'_> {}

impl BackendExecutionContext for ONNXExecutionContext<'_> {
    fn set_input(&mut self, index: u32, tensor: &Tensor) -> Result<(), BackendError> {
        self.inputs[index as usize].replace(tensor.clone());
        Ok(())
    }

    fn compute(&mut self) -> Result<(), BackendError> {
        let shaped_inputs = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, _o)| {
                let input = self.inputs[i].as_ref().unwrap();
                let dims = input
                    .dimensions
                    .as_slice()
                    .iter()
                    .map(|d| *d as usize)
                    .collect::<Vec<_>>();
                println!("input dims: {:?}", dims);
                match input.tensor_type {
                    TensorType::Fp32 => {
                        let data = bytes_to_f32_vec(input.data.to_vec());
                        Array::from_shape_vec(dims, data).unwrap()
                    }
                    TensorType::Fp16 => {
                        unimplemented!("Fp16 not supported by ONNX");
                    }
                    TensorType::Bf16 => {
                        unimplemented!("Bf16 not supported by ONNX");
                    }
                    TensorType::U8 => {
                        unimplemented!("U8 not supported by ONNX");
                        // Array::from_shape_vec(dims, input.data.to_vec()).unwrap()
                    }
                    TensorType::I32 => {
                        unimplemented!("I32 not supported by ONNX");
                        // Array::from_shape_vec(dims, input.data.to_vec()).unwrap()
                    }
                }
            })
            .collect();

        let mut session = self.session.lock().unwrap();
        let res: Vec<OrtOwnedTensor<f32, _>> = session.run(shaped_inputs)?;

        for i in 0..self.outputs.len() {
            self.outputs[i].replace(f32_vec_to_bytes(res[i].to_slice().unwrap().to_vec()));
        }
        Ok(())
    }

    fn get_output(&mut self, index: u32, destination: &mut [u8]) -> Result<u32, BackendError> {
        let output = self.outputs[index as usize].as_ref().unwrap();
        destination[..output.len()].copy_from_slice(output);
        Ok(output.len() as u32)
    }
}

impl From<OrtError> for BackendError {
    fn from(e: OrtError) -> Self {
        BackendError::BackendAccess(e.into())
    }
}

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<f32> = chunks
        .into_iter()
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    v.into_iter().collect()
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let result: Vec<u8> = chunks.iter().flatten().copied().collect();
    result
}
