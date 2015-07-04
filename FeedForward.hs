module FeedForward ( NeuralNetwork
                   , WeightMatrix
                   , ActivationFunction
                   , sigmoid
                   , sigmoid'
                   , dim
                   , getStructure
                   , getWeights
                   , getActivationFunction
                   , generateRandomWeights
                   , generateRandomWeights_
                   , mkNeuralNetwork
                   , mkRandomNeuralNetwork
                   , mkRandomNeuralNetwork_
                   ) where

import Data.Vector (Vector(..))
import qualified Data.Vector as V
import System.Random

-- | Activation function for neurons in a layer
type ActivationFunction = Double -> Double

-- | Sigmoid function, has a nice derivative
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

-- | Sigmoid derivative, expressed nicely in function of the sigmoid
sigmoid' :: Double -> Double
sigmoid' x = sigmoid (x) * (1 - sigmoid x)

-- Matrix of weights between two layers
type WeightMatrix = Vector (Vector Double)

-- | Data type representing a Feed-Forward Neural Network
data NeuralNetwork = NeuralNetwork
                     { structure :: [Int]   -- ^ [3,4,2,1] for a 3-layered network, with 3 input units, 1 output unit, 4 units in first hidden layer and 2 units in second hidden layer
                     , weights      :: [WeightMatrix] -- ^ Weights matrices to apply from each layer to the following
                     , activationFunction :: Double -> Double
                     }

-- | Returns a Neural Network's architecture (since internal structure is not exposed)
getStructure :: NeuralNetwork -> [Int]
getStructure (NeuralNetwork {structure = s}) = s

-- | Returns a Neural Network's weight matrix (since internal structure is not exposed)
getWeights :: NeuralNetwork -> [WeightMatrix]
getWeights (NeuralNetwork {weights = w}) = w

-- | Returns a Neural Network's activation function (since internal structure is not exposed)
getActivationFunction :: NeuralNetwork -> (Double -> Double)
getActivationFunction (NeuralNetwork {activationFunction = h}) = h

-- | Returns the dimensions of a weight matrix (nbLines, nbColumns)
dim :: WeightMatrix -> (Int, Int)
dim lines = (V.length lines, V.length . V.head $ lines)

data NNError = StructureError String
             | OtherError String
             deriving (Show)

-- | Create a Feed-Forward Neural Network, ensuring weight matrix dimensions fit
mkNeuralNetwork :: ActivationFunction -> [Int] -> [WeightMatrix] -> Either NNError NeuralNetwork
mkNeuralNetwork _ [] _ = Left $ StructureError "Structure is empty"
mkNeuralNetwork _ _ [] = Left $ StructureError "Weight matrix is empty"
mkNeuralNetwork _ [_] _ = Left $ StructureError "A neural network must have at least one input and one output layer"
mkNeuralNetwork _ _ [_] = Left $ StructureError "A neural network must have at least one weight matrix (from input to second layer)"
mkNeuralNetwork h layers weights | length layers /= length weights + 1 = Left $ StructureError "For a K-layered Neural Network, you must have K-1 weight matrices"
                               | checkDims layers weights == True = Right $ NeuralNetwork
                                                                    { structure = layers
                                                                    , weights = weights
                                                                    , activationFunction = h
                                                                    }
                               | otherwise = Left $ StructureError "Weight matrix dimension not compatible with architecture. Matrix dimension for layer of size M to layer of size N must be (M, N)"
  where checkDims :: [Int] -> [WeightMatrix] -> Bool
        checkDims [lastLayer] [] = True
        checkDims _ [] = False
        checkDims (l1:l2:ls) (w1:ws) | dim w1 == (l1, l2) = checkDims (l2:ls) ws
                                     | otherwise          = False
        checkDims _ _ = False

-- | Create a Neural Network, whose weights are randomly initialized, given a range
mkRandomNeuralNetwork :: RandomGen g => g -> ActivationFunction -> [Int] -> (Double, Double) -> Either NNError NeuralNetwork
mkRandomNeuralNetwork _ h [] _ = mkNeuralNetwork h [] []
mkRandomNeuralNetwork gen h xs (lowerBound, upperBound) =
  let randomWeights = generateRandomWeights gen xs (lowerBound, upperBound)
  in mkNeuralNetwork h xs randomWeights

-- IO Version of `mkRandomNeuralNetwork_`
mkRandomNeuralNetwork_ :: ActivationFunction -> [Int] -> (Double, Double) -> IO (Either NNError NeuralNetwork)
mkRandomNeuralNetwork_ h [] _ = return $ mkNeuralNetwork h [] []
mkRandomNeuralNetwork_ h xs (lowerBound, upperBound) = do
  randomWeights <- generateRandomWeights_ xs (lowerBound, upperBound)
  return $ mkNeuralNetwork h xs randomWeights
  
generateRandomLine :: [Double] -> Int -> (Vector Double, [Double])
generateRandomLine pool n =
  let (values, rest) = splitAt n pool
  in (V.fromList values, rest)

generateNRandomLines :: [Double] -> Int -> Int -> [Vector Double]
generateNRandomLines pool 0 _ = []
generateNRandomLines pool n dim =
  let (oneLine, restOfPool) = generateRandomLine pool dim
  in oneLine : generateNRandomLines restOfPool (n-1) dim

generateRandomMatrix :: [Double] -> (Int, Int) -> Vector (Vector Double)
generateRandomMatrix pool (rows, cols) = V.fromList $ generateNRandomLines pool rows cols

generateRandomWeights :: RandomGen g => g -> [Int] -> (Double, Double) -> [WeightMatrix]
generateRandomWeights gen xs range =
  let pool = randomRs range gen
  in [generateRandomMatrix pool (l1, l2) | (l1, l2) <- zip xs (tail xs)]

generateRandomWeights_ :: [Int] -> (Double, Double) -> IO [WeightMatrix]
generateRandomWeights_ xs range = do
  pool <- getStdGen >>= return . randomRs range
  return $ [generateRandomMatrix pool (l1, l2) | (l1, l2) <- zip xs (tail xs)]

-- Feed the Neural Network with data and get back results
forwardPass :: NeuralNetwork -> [Vector Double] -> [Vector Double]
forwardPass _ [] = []
forwardPass nn (x:xs) =
  let f = activationFunction nn
      w = weights nn
