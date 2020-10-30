//! This crate is a Rust implementation of a toy neural network
//! library, which allows first steps in AI and machine learning
//! in the browser with JavaScript. Its easy to create, train,
//! test, use, save and load a neural network. It also exposes
//! basic functionality of linear algebra with matrices. With the
//! power of WebAssembly, it's very fast compared to a pure JS
//! implementation.

mod matrix;
mod neural_net;

pub use matrix::Matrix;
pub use neural_net::NeuralNet;
