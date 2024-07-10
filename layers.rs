use neural_net_core::tensor::Tensor;
use neural_net_core::activation::*;
use neural_net_core::operations::{matmul, add};

/// A fully connected layer.
pub struct FullyConnected {
    pub weights: Tensor,
    pub biases: Tensor,
}

impl FullyConnected {
    pub fn new(input_size: usize, output_size: usize) -> FullyConnected {
        FullyConnected {
            weights: Tensor::random(input_size, output_size),
            biases: Tensor::zero(1, output_size),
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        let z = matmul(input, &self.weights) + &self.biases;
        relu(&z)
    }
}

/// A convolutional layer.
pub struct Convolution {
    pub filters: Tensor,
    pub biases: Tensor,
    pub stride: usize,
    pub padding: usize,
}

impl Convolution {
    pub fn new(
        input_channels: usize,
        output_channels: usize,
        filter_size: usize,
        stride: usize,
        padding: usize,
    ) -> Convolution {
        Convolution {
            filters: Tensor::random(output_channels, input_channels, filter_size, filter_size),
            biases: Tensor::zero(1, output_channels, 1, 1),
            stride,
            padding,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Implement the convolutional forward pass here
        unimplemented!()
    }
}

/// A max pooling layer.
pub struct MaxPooling {
    pub pool_size: usize,
    pub stride: usize,
}

impl MaxPooling {
    pub fn new(pool_size: usize, stride: usize) -> MaxPooling {
        MaxPooling { pool_size, stride }
    }

    pub fn forward(&self, input: &Tensor) -> Tensor {
        // Implement the max pooling forward pass here
        unimplemented!()
    }
}