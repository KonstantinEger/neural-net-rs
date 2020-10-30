use wasm_bindgen::prelude::wasm_bindgen;

/// A matrix is like a table of `f64` numbers. Each item has a position
/// and value.
#[wasm_bindgen]
pub struct Matrix
{
	rows: u32,
	cols: u32,
	data: Vec<f64>,
}

/// Methods in this `impl` are shared and accessable from JavaScript.
#[wasm_bindgen]
impl Matrix
{
	/// Returns a new instance of a matrix. It's initialized with
	/// all `0f64`.
	#[wasm_bindgen(constructor)]
	pub fn new(rows: u32, cols: u32) -> Self
	{
		Self {
			rows, cols,
			data: vec![0f64; (rows * cols) as usize]
		}
	}

	/// Returns the amount of rows the matrix has.
	/// ```
	/// use neural_net_rs::Matrix;
	/// let m = Matrix::new(2, 3);
	/// assert_eq!(2, m.rows());
	/// ```
	pub fn rows(&self) -> u32
	{
		self.rows
	}

	/// Returns the amount of columns the matrix has.
	/// ```
	/// use neural_net_rs::Matrix;
	/// let m = Matrix::new(2, 3);
	/// assert_eq!(3, m.cols());
	/// ```
	pub fn cols(&self) -> u32
	{
		self.cols
	}

	/// Returns a one-dimensional representation of the data inside
	/// the matrix.
	/// ```
	/// use neural_net_rs::Matrix;
	/// let mut m = Matrix::new(2, 3);
	/// m.map(|_, r, c| (r + c) as f64);
	/// let should_be = vec![
	/// 	0., 1., 2.,
	/// 	1., 2., 3.
	/// ];
	/// assert_eq!(m.data(), should_be);
	/// ```
	pub fn data(&self) -> Vec<f64>
	{
		self.data.clone()
	}

	/// Scale every item in the matrix by some float.
	/// ```
	/// let mut m = neural_net_rs::Matrix::new(2, 3);
	/// m.map(|_, r, c| (r + c) as f64);
	/// 
	/// let before_scale = vec![
	/// 	0., 1., 2.,
	/// 	1., 2., 3.
	/// ];
	/// let after_scale = vec![
	/// 	0., 3., 6.,
	/// 	3., 6., 9.
	/// ];
	/// assert_eq!(m.data(), before_scale);
	/// m.scale(3.);
	/// assert_eq!(m.data(), after_scale);
	/// ```
	pub fn scale(&mut self, num: f64)
	{
		self.map(|val, _, _| val * num);
	}

	/// Converts a 2D position in the matrix into a index to
	/// look up in the data array.
	fn calc_idx(&self, i: u32, j: u32) -> usize
	{
		(i * self.cols + j) as usize
	}
}

/// Methods in this `impl` are **not** accessable from JavaScript.
impl Matrix {
	/// Maps over each position in the matrix (starting top-left
	/// going left to right). A callback is called on each item
	/// of the matrix with the current `value`, its `row` and
	/// its `column`.
	pub fn map<F>(&mut self, cb: F) -> ()
	where F: Fn(f64, u32, u32) -> f64
	{
		for i in 0..self.rows {
			for j in 0..self.cols {
				let idx = self.calc_idx(i, j);
				self.data[idx] = cb(self.data[idx], i, j);
			}
		}
	}
}
