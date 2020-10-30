use wasm_bindgen::prelude::wasm_bindgen;

/// An instance of NeuralNet is able to perform calculations on some
/// input data. It can be "trained" to give a specific result on some
/// specific input. It consists of layers of nodes, through which data
/// "flows".
#[wasm_bindgen]
pub struct NeuralNet {}

#[wasm_bindgen]
impl NeuralNet
{
	/// Returns a new instance of a NeuralNet.
	#[wasm_bindgen(constructor)]
	pub fn new() -> Self
	{
		#[cfg(feature = "console_error_panic_hook")]
		console_error_panic_hook::set_once();

		Self {}
	}
}
