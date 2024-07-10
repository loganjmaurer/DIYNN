module Neural.Net.Core.Backpropagation (
  Backpropagation(..),
  train
) where

import Neural.Net.Core.Layers
import Neural.Net.Core.Loss
import Neural.Net.Core.Tensor
import Neural.Net.Core.Activation

-- | A backpropagation trainer
data Backpropagation = Backpropagation
  { learningRate :: Double
  , lossFunction :: Loss a
  }

-- | Train a neural network using backpropagation
train :: Backpropagation
      -> [Tensor] -- ^ Input data
      -> [Tensor] -- ^ Target output
      -> [(Tensor, Tensor)] -- ^ Trained weights and biases
train bp inputs targets = go inputs targets [] []
  where
    go [] [] weights biases = zip weights biases
    go (input:inputs) (target:targets) weights biases = do
      let (output, layer_outputs) = feedforward input weights biases
          loss = lossFunction bp output target
          grads = backpropagate loss layer_outputs
          new_weights = updateWeights weights grads
          new_biases = updateBiases biases grads
      go inputs targets new_weights new_biases

-- | Perform a forward pass through the network
feedforward :: Tensor -> [(Tensor, Tensor)] -> [(Tensor, Tensor)] -> (Tensor, [(Tensor, Tensor)])
feedforward input weights biases = foldl' step (input, []) $ zip weights biases
  where
    step (a, history) (w, b) = do
      let z = matmul a w + b
          a' = relu z
      (a', (w, b, z, a') : history)

-- | Backpropagate the gradients through the network
backpropagate :: Loss a -> [(Tensor, Tensor, Tensor, Tensor)] -> [(Tensor, Tensor)]
backpropagate loss layers = reverse $ foldl' step [] layers
  where
    step grads (w, b, z, a) = do
      let da = gradients loss a
          dw = matmul (transpose a) da
          db = sum da
      ((w, dw), (b, db)) : grads

-- | Update the weights and biases using the gradients
updateWeights :: [(Tensor, Tensor)] -> [(Tensor, Tensor)] -> [(Tensor, Tensor)]
updateWeights weights grads = zipWith (\(w, b) (dw, _) -> (w - learningRate * dw, b)) weights grads

updateBiases :: [(Tensor, Tensor)] -> [(Tensor, Tensor)] -> [(Tensor, Tensor)]
updateBiases biases grads = zipWith (\(w, b) (_, db) -> (w, b - learningRate * db)) biases grads