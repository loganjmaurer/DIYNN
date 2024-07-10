use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::ops::{Add, Mul, Sub};

/// Represents a multi-dimensional tensor.
pub struct Tensor(ArrayD<f32>);

impl Tensor {
    /// Create a new tensor from a multi-dimensional array.
    pub fn new(data: ArrayD<f32>) -> Self {
        Tensor(data)
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.0.shape()
    }

    /// Get the number of dimensions of the tensor.
    pub fn ndim(&self) -> usize {
        self.0.ndim()
    }

    /// Reshape the tensor to the given shape.
    pub fn reshape(&mut self, new_shape: &[usize]) -> &mut Self {
        self.0.into_shape(new_shape).unwrap();
        self
    }

    /// Transpose the tensor.
    pub fn transpose(&mut self) -> &mut Self {
        self.0.swap_axes(0, 1);
        self
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        Tensor(self.0 + other.0)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        Tensor(self.0 - other.0)
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        Tensor(self.0 * other.0)
    }
}