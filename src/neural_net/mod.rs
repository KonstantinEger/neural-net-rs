use wasm_bindgen::prelude::wasm_bindgen;
use crate::Matrix;

/// An instance of NeuralNet is able to perform calculations on some
/// input data. It can be "trained" to give a specific result on some
/// specific input. It consists of layers of nodes, through which data
/// "flows".
#[wasm_bindgen]
pub struct NeuralNet {
	hidden_nodes: Vec<u32>,
	hidden_weights: Vec<Matrix>,
	learning_rate: f64,
	bias: u8,
}

#[wasm_bindgen]
impl NeuralNet
{
	/// Returns a new instance of a `NeuralNet`. The first argument
	/// is the size of the input layer. The second arg is a `vec`
	/// where each item represents the size of one hidden layer.
	/// An empty vector can be supplied to create an `Perceptron`.
	/// The third argument is the size of the output layer.
	/// ```
	/// use neural_net_rs::NeuralNet;
	/// let nn = NeuralNet::new(3, vec![2, 3], 2);
	/// ```
	/// This `Neural Network` would consist of an input layer with
	/// `3` nodes, a hidden layer with `2`, one with `3` nodes
	/// and an output layer with `2` nodes.
	#[wasm_bindgen(constructor)]
	pub fn new(input_nodes: u32, hidden_nodes: Vec<u32>, output_nodes: u32) -> NeuralNet
	{
		#[cfg(feature = "console_error_panic_hook")]
		console_error_panic_hook::set_once();

		let hn_len = hidden_nodes.len();
		let mut hidden_weights: Vec<Matrix> = Vec::new();

		if hn_len > 0 {
			hidden_weights.push(Matrix::new(hidden_nodes[0], input_nodes));

			for i in 1..hn_len {
				hidden_weights.push(Matrix::new(hidden_nodes[i], hidden_nodes[i-1]));
			}

			hidden_weights.push(Matrix::new(output_nodes, hidden_nodes[hn_len - 1]));
		} else {
			hidden_weights.push(Matrix::new(output_nodes, input_nodes));
		}


		NeuralNet {
			learning_rate: 0.1_f64,
			bias: 1,
			hidden_nodes,
			hidden_weights
		}
	}
}

#[cfg(test)]
mod tests
{
	use super::NeuralNet;

	#[test]
	fn nn_new()
	{
		let nn = NeuralNet::new(2, vec![3, 4, 5], 2);
		assert_eq!(nn.hidden_weights.len(), 4);
	}

	#[test]
	fn new_perceptron()
	{
		let nn = NeuralNet::new(2, Vec::new(), 1);
		assert_eq!(nn.hidden_weights.len(), 1);
	}
}
