use neural_net_core::tensor::Tensor;
use neural_net_core::operations::{matmul, mean};

/// Performs a gradient descent update step for the given parameters and gradients.
pub fn gradient_descent(params: &mut [Tensor], grads: &[Tensor], learning_rate: f64) {
    for (param, grad) in params.iter_mut().zip(grads) {
        param.0 -= &(grad.0 * learning_rate);
    }
}

/// Computes the mean squared error loss between the predicted and target tensors.
pub fn mean_squared_error(predicted: &Tensor, target: &Tensor) -> Tensor {
    let diff = &predicted.0 - &target.0;
    Tensor(mean(&Tensor(diff * diff), Some(Axis(0))))
}

/// Computes the cross-entropy loss between the predicted and target tensors.
pub fn cross_entropy(predicted: &Tensor, target: &Tensor) -> Tensor {
    let log_predicted = Tensor(predicted.0.map(|x| x.ln()));
    Tensor(-mean(&Tensor(target.0 * log_predicted.0), Some(Axis(0))))
}

/// Computes the gradients of the mean squared error loss with respect to the given parameters.
pub fn mse_gradients(params: &[Tensor], predicted: &Tensor, target: &Tensor) -> Vec<Tensor> {
    let mut gradients = Vec::with_capacity(params.len());

    // Compute the gradient for each parameter
    for param in params {
        let diff = &predicted.0 - &target.0;
        let grad = 2.0 * mean(&Tensor(diff * &param.0), Some(Axis(0))).0;
        gradients.push(Tensor(grad));
    }

    gradients
}

/// Computes the gradients of the cross-entropy loss with respect to the given parameters.
pub fn ce_gradients(params: &[Tensor], predicted: &Tensor, target: &Tensor) -> Vec<Tensor> {
    let mut gradients = Vec::with_capacity(params.len());

    // Compute the gradient for each parameter
    for param in params {
        let grad = -mean(&Tensor(target.0 * &param.0), Some(Axis(0))).0;
        gradients.push(Tensor(grad));
    }

    gradients
}