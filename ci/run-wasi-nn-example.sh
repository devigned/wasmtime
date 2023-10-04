#!/bin/bash

# The following script demonstrates how to execute a machine learning inference
# using the wasi-nn module optionally compiled into Wasmtime. Calling it will
# download the necessary model and tensor files stored separately in $FIXTURE
# into $TMP_DIR (optionally pass a directory with existing files as the first
# argument to re-try the script). Then, it will compile and run several examples
# in the Wasmtime CLI.
set -e
WASMTIME_DIR=$(dirname "$0" | xargs dirname)
FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
if [ -z "${1+x}" ]; then
    # If no temporary directory is specified, create one.
    TMP_DIR=$(mktemp -d -t ci-XXXXXXXXXX)
    REMOVE_TMP_DIR=1
else
    # If a directory was specified, use it and avoid removing it.
    TMP_DIR=$(realpath $1)
    REMOVE_TMP_DIR=0
fi

# One of the examples expects to be in a specifically-named directory.
mkdir -p $TMP_DIR/mobilenet
TMP_DIR=$TMP_DIR/mobilenet

# Set up the environment for OpenVINO on OSX. This is necessary because on OSX, DYLD_LIBRARY_PATH will be ignored when
# running this file as a script. See: https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_security_cs_allow-dyld-environment-variables?language=objc
export DYLD_LIBRARY_PATH="${OSX_OPENVINO_LIB_PATH}"

# Build Wasmtime with wasi-nn enabled; we attempt this first to avoid extra work
# if the build fails.
cargo install --git https://github.com/bytecodealliance/cargo-component --locked cargo-component
cargo build -p wasmtime-cli --features wasi-nn,component-model

# Download all necessary test fixtures to the temporary directory.
wget --no-clobber $FIXTURE/mobilenet.bin --output-document=$TMP_DIR/model.bin
wget --no-clobber $FIXTURE/mobilenet.xml --output-document=$TMP_DIR/model.xml
wget --no-clobber $FIXTURE/tensor-1x224x224x3-f32.bgr --output-document=$TMP_DIR/tensor.bgr

# Build and run an ONNX component example doing classification.
pushd $WASMTIME_DIR/crates/wasi-nn/examples/classification-component-onnx
cargo component build --release
cp target/wasm32-wasi/release/classification-component-onnx.wasm $TMP_DIR
popd
cargo run --features wasi-nn,component-model -- run --wasm component-model --dir fixture::$TMP_DIR -S nn $TMP_DIR/classification-component-onnx.wasm


# Now build an example that uses the wasi-nn API. Run the example in Wasmtime
# (note that the example uses `fixture` as the expected location of the
# model/tensor files).
pushd $WASMTIME_DIR/crates/wasi-nn/examples/classification-example
cargo build --release --target=wasm32-wasi
cp target/wasm32-wasi/release/wasi-nn-example.wasm $TMP_DIR
popd

cargo run -- run --dir fixture::$TMP_DIR -S nn $TMP_DIR/wasi-nn-example.wasm

# Build and run another example, this time using Wasmtime's graph flag to
# preload the model.
pushd $WASMTIME_DIR/crates/wasi-nn/examples/classification-example-named
cargo build --release --target=wasm32-wasi
cp target/wasm32-wasi/release/wasi-nn-example-named.wasm $TMP_DIR
popd
cargo run -- run --dir fixture::$TMP_DIR -S nn,nn-graph=openvino::$TMP_DIR \
    $TMP_DIR/wasi-nn-example-named.wasm

# Clean up the temporary directory only if it was not specified (users may want
# to keep the directory around).
if [[ $REMOVE_TMP_DIR -eq 1 ]]; then
    rm -rf $TMP_DIR
fi
