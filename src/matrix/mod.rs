use wasm_bindgen::prelude::{JsValue, wasm_bindgen };
use std::convert::TryInto;

/// A matrix is like a table of `f64` numbers. Each item has a position
/// and value.
#[wasm_bindgen]
#[derive(Clone)]
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
	/// Returns a new instance of a matrix: `rows`x`cols`. It's initialized with
	/// all `0f64`.
	#[wasm_bindgen(constructor)]
	pub fn new(rows: u32, cols: u32) -> Self
	{
		Self {
			rows, cols,
			data: vec![0_f64; (rows * cols) as usize]
		}
	}

	/// Turns a vector into a matrix.
	/// ```
	/// let v = vec![1., 2., 3., 4., 5., 6.];
	/// let m1 = neural_net_rs::Matrix::from(2, 3, v.clone()).unwrap();
	/// let m2 = neural_net_rs::Matrix::from(3, 2, v).unwrap();
	/// assert_eq!(m1.data(), m2.data());
	/// // |1, 2, 3|		|1, 2|
	/// // |4, 5, 6|	or	|3, 4|
	/// //					|5, 6|
	/// ```
	pub fn from(rows: u32, cols: u32, list: Vec<f64>) -> Result<Matrix, JsValue>
	{
		if rows * cols != list.len().try_into().unwrap() {
			return Err(JsValue::from_str("Length of list does not match `rows` x `cols`"))
		}

		let mut result = Matrix::new(rows, cols);
		for (idx, val) in list.iter().enumerate() {
			result.data[idx] = *val;
		}

		Ok(result)
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

	/// Returns the value at a row & column position of the matrix.
	/// ```
	/// let mut m = neural_net_rs::Matrix::new(2, 2);
	/// let m_cols = m.cols();
	/// m.map(|_, row, col| ((row * m_cols + col) + 1) as f64);	// |1, 2|
	/// assert_eq!(m.data(), vec![1., 2., 3., 4.]);				// |3, 4|
	/// assert_eq!(m.get(0, 0), 1.);
	/// assert_eq!(m.get(0, 1), 2.);
	/// assert_eq!(m.get(1, 0), 3.);
	/// assert_eq!(m.get(1, 1), 4.);
	/// ```
	pub fn get(&self, row: u32, col: u32) -> f64
	{
		let idx = self.calc_idx(row, col);
		*(&self.data[idx])
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

	/// Matrix product of two matrices -> returns a new Matrix.
	/// Could fail because because columns of `a` (self) must match
	/// rows of `b` (other).
	/// ```
	/// let a = neural_net_rs::Matrix::from(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
	/// let b = neural_net_rs::Matrix::from(3, 2, vec![7., 8., 9., 10., 11., 12.]).unwrap();
	/// // |1, 2, 3|   | 7,  8|
	/// // |4, 5, 6| X | 9, 10|
	/// //             |11, 12|
	/// let c = neural_net_rs::Matrix::mult(&a, &b).unwrap();
	/// assert_eq!(c.rows(), 2);
	/// assert_eq!(c.cols(), 2);
	/// assert_eq!(c.data(), vec![58., 64., 139., 154.]);
	/// ```
	pub fn mult(a: &Matrix, b: &Matrix) -> Result<Matrix, JsValue>
	{
		if a.cols() != b.rows() {
			return Err(JsValue::from_str("Error: columns of left-hand-side must match rows of right-hand-side"));
		}

		let mut result = Matrix::new(a.rows(), b.cols());

		result.map(|_, row, col| {
			let mut sum = 0_f64;
			for k in 0..a.cols() {
				sum = sum + a.get(row, k) * b.get(k, col);
			}
			sum
		});
		
		Ok(result)
	}
}

/// Methods in this `impl` are **not** accessable from JavaScript.
impl Matrix
{
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

#[cfg(test)]
mod tests
{
	use super::Matrix;

	#[test]
	fn matrix_new()
	{
		let m = Matrix::new(3, 4);
		assert_eq!(m.rows(), 3);
		assert_eq!(m.cols(), 4);
		assert_eq!(m.data(), vec![0_f64; 12]);
	}

	#[test]
	fn matrix_from()
	{
		let v = vec![1., 2., 3., 4., 5., 6.];
		let m1 = Matrix::from(2, 3, v.clone()).unwrap();
		let m2 = Matrix::from(3, 2, v.clone()).unwrap();
		assert_eq!(m1.rows(), 2);
		assert_eq!(m1.cols(), 3);
		assert_eq!(m1.data(), v);
		assert_eq!(m2.data(), v);
	}

	#[test]
	fn getters()
	{
		let mut m = Matrix::new(2, 3);
		m.map(|_, r, c| (r + c) as f64);
		assert_eq!(m.rows(), 2); // |0, 1, 2|
		assert_eq!(m.cols(), 3); // |1, 2, 3|
		assert_eq!(m.data(), vec![0., 1., 2., 1., 2., 3.]);
		assert_eq!(m.get(0, 1), 1.);
		assert_eq!(m.get(1, 1), 2.);
	}

	#[test]
	fn scale()
	{
		let mut m = Matrix::from(2, 2, vec![1., 2., 3., 4.]).unwrap();
		m.scale(2.);
		assert_eq!(m.data(), vec![2., 4., 6., 8.]);
	}

	#[test]
	fn calc_idx()
	{
		let m = Matrix::new(3, 4);
		// |0, 1,  2,  3|
		// |4, 5,  6,  7|
		// |8, 9, 10, 12|
		assert_eq!(m.calc_idx(0, 0), 0);
		assert_eq!(m.calc_idx(0, 2), 2);
		assert_eq!(m.calc_idx(1, 1), 5);
		assert_eq!(m.calc_idx(2, 2), 10);
	}

	#[test]
	fn mult()
	{
		let a = Matrix::from(2, 3, vec![5., 1., 2., 3., 2., 6.]).unwrap();
		let b = Matrix::from(3, 2, vec![8., 7., 4., 4., 5., 1.]).unwrap();

		let c = Matrix::mult(&a, &b).unwrap();
		assert_eq!(c.rows(), 2);
		assert_eq!(c.cols(), 2);
		assert_eq!(c.data(), vec![54., 41., 62., 35.]);
	}

	#[test]
	fn map()
	{
		let mut m = Matrix::from(2, 3, vec![1., 2., 3., 4., 5., 6.]).unwrap();
		m.map(|val, r, c| (val * (c * r) as f64));
		assert_eq!(m.data(), vec![0., 0., 0., 0., 5., 12.]);
	}
}