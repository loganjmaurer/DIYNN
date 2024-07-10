{-# LANGUAGE OverloadedLists #-}

module Neural.Net.Model (
  Model(..),
  buildModel,
  trainModel
) where

import Neural.Net.Core.Layers
import Neural.Net.Core.Loss
import Neural.Net.Core.Tensor
import Neural.Net.Core.Activation
import Neural.Net.Core.Backpropagation

-- | A neural network model
data Model = Model
  { layers :: [Layer]
  , lossFunction :: Loss a
  , backpropagation :: Backpropagation
  }

-- | Build a neural network model
buildModel :: [Layer] -> Loss a -> Double -> Model
buildModel layers lossFunction learningRate = Model
  { layers = layers
  , lossFunction = lossFunction
  , backpropagation = Backpropagation
    { learningRate = learningRate
    , lossFunction = lossFunction
    }
  }

-- | Train a neural network model
trainModel :: Model -> [Tensor] -> [Tensor] -> [(Tensor, Tensor)]
trainModel model inputs targets = train (backpropagation model) inputs targets
  where
    (weights, biases) = zip (map weights $ layers model) (map biases $ layers model)

-- | Example usage
example :: IO ()
example = do
  let layers = [ Dense 10 relu
               , Dense 5 relu
               , Dense 1 sigmoid
               ]
      model = buildModel layers categoricalCrossEntropy 0.01
      inputs = ... -- input data
      targets = ... -- target output
  trainedWeightsAndBiases <- trainModel model inputs targets
  print trainedWeightsAndBiases