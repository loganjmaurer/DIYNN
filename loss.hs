module Neural.Net.Core.Loss (
  Loss(..),
  meanSquaredError,
  binaryCrossEntropy,
  categoricalCrossEntropy
) where

import Neural.Net.Core.Tensor (Tensor)

-- | A typeclass for loss functions
class Loss a where
  -- | Compute the loss between the predicted output and the true output
  loss :: Tensor -> Tensor -> a

-- | Mean Squared Error (MSE) loss
data MeanSquaredError = MeanSquaredError
instance Loss MeanSquaredError where
  loss pred true = sum $ (pred - true) ^^ 2 / (fromIntegral $ length $ flatten pred)

-- | Binary Cross-Entropy (BCE) loss
data BinaryCrossEntropy = BinaryCrossEntropy
instance Loss BinaryCrossEntropy where
  loss pred true = - sum $ (true * log pred) + ((1 - true) * log (1 - pred))

-- | Categorical Cross-Entropy (CCE) loss
data CategoricalCrossEntropy = CategoricalCrossEntropy
instance Loss CategoricalCrossEntropy where
  loss pred true = - sum $ true * log pred

-- | Convenience functions for the loss functions
meanSquaredError :: Tensor -> Tensor -> MeanSquaredError
meanSquaredError = loss

binaryCrossEntropy :: Tensor -> Tensor -> BinaryCrossEntropy
binaryCrossEntropy = loss

categoricalCrossEntropy :: Tensor -> Tensor -> CategoricalCrossEntropy
categoricalCrossEntropy = loss