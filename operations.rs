use ndarray::{Array, ArrayD, Axis};
use neural_net_core::tensor::Tensor;

/// Computes the matrix multiplication of two tensors.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor(a.0.dot(&b.0))
}

/// Applies the ReLU (Rectified Linear Unit) activation function element-wise to the input tensor.
pub fn relu(x: &Tensor) -> Tensor {
    Tensor(x.0.map(|x| x.max(0.0)))
}

/// Computes the mean of the elements in the input tensor along the given axis.
pub fn mean(x: &Tensor, axis: Option<Axis>) -> Tensor {
    Tensor(x.0.mean_axis(axis))
}

/// Computes the softmax of the input tensor along the given axis.
pub fn softmax(x: &Tensor, axis: Axis) -> Tensor {
    let exp = x.0.exp();
    let sum = exp.sum_axis(axis);
    Tensor(exp / sum.insert_axis(axis))
}