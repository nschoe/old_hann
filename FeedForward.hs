{-# LANGUAGE BangPatterns #-}

module FeedForward ( NeuralNetwork
                   , WeightMatrix
                   , ActivationFunction
                   , LearningRateStrategy (..)
                   , BackPropStrategy (..)
                   , sigmoid
                   , sigmoid'
                   , getStructure
                   , getWeights
                   , getActivationFunction
                   , mkNeuralNetwork
                   , test
                   ) where

import           Data.List (foldl')
import           Data.Vector (Vector(..))
import qualified Data.Vector as V (singleton, scanl, scanr, fromList, toList, zip, last, tail, init, zipWith3)
import           Debug.Trace (trace)
import           Numeric.LinearAlgebra.HMatrix
import           System.Random (randomRs, newStdGen)

-- | Activation function for neurons in a layer
type ActivationFunction = Double -> Double

-- | Sigmoid function, has a nice derivative
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))

-- | Sigmoid derivative, expressed nicely in function of the sigmoid
sigmoid' :: Double -> Double
sigmoid' x = sigmoid (x) * (1 - sigmoid x)

-- Matrix of weights between two layers
type WeightMatrix = Matrix Double

-- List of number of units in each layer, first layer is input, last is output
type Structure = [Int]

-- Learning rate strategy for training
data LearningRateStrategy = FixedRate Double -- Learning rate alpha will remain constant
                          deriving (Show, Eq)

-- | The Gradient Descent Strategy to use with BackPropagation
data BackPropStrategy = BatchGradientDescent         -- Accumulate error on all cases before performing a weights update
                      | MiniBatchGradientDescent Int -- Update weights after N computations
                      | OnlineGradientDescent        -- a.k.a "stochastic", this is mini batch with N=1 : update weights after each training case
                        deriving (Show, Eq)

-- | Type that defines a training example: a pair of the input vector and the target vector
type TrainingExample = (Matrix Double, Matrix Double) -- Must be column vector

-- | The training data set is the list of all training examples
type TrainingSet = [TrainingExample]

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

{- | Create a Feed-Forward Neural Network, whose weights are randomly initialized following this procedure:
    - first, random weights are generated in the range (-1/√i, +1/√i) where i is the number of input neurons
    - then use Nguyen Widrow method to readjust the weights distribution

    _Note_: Biais neurons are automatically added to each layer, so do not consider them

    _Note_: for now, the method is not perfect because it only computes h, i  and n for the irst layer, it should be done for all layers to be accurate. **TO FIX**
-}
mkNeuralNetwork :: ActivationFunction -> Structure -> IO NeuralNetwork
mkNeuralNetwork _ xs | length xs < 2 = error "A Neural Network must have at least one input layer and an ouput layer, so your structure must contain at least 2 numbers"
                     | any (<= 0) xs = error "You can't have zero or a negative number of units in a layer"
mkNeuralNetwork h xs = do
  --let bound = 1.0 / sqrt i :: Double
  let bound = 10.0 / sqrt i :: Double
  initialRandomWeights <- newStdGen >>= return . randomRs (-bound, bound)
  let !weightMatrices = shapeWeightMatrices initialRandomWeights xs
  putStrLn . show $ weightMatrices
  return $ NeuralNetwork
    { structure          = xs
    , weights            = weightMatrices
    , activationFunction = sigmoid
    }
{-
  let beta = 0.7 * (h ** (1.0 / i))
      n = sqrt (sum
-}
        where h = fromIntegral (xs !! 1) :: Double -- nb of hidden neurons
              i = fromIntegral . head $ xs :: Double -- nb of input features

              shapeWeightMatrices :: [Double] -> Structure -> [WeightMatrix]
              shapeWeightMatrices pool [_] = []
              shapeWeightMatrices pool (l1:l2:xs) =
                let l1' = l1 + 1
                in (l1'><l2) pool : shapeWeightMatrices (drop (l1'*l2) pool) (l2:xs)

-- Run the Neural Network on the input matrix to get output matrix (automatically add biais neurons with value 1)
runNN :: NeuralNetwork -> Matrix Double -> Matrix Double
runNN nn input =
  let ws = weights nn
      h  = activationFunction nn
  in foldl' addOnesAndMultiply input ws
     
  where addOnesAndMultiply :: Matrix Double -> Matrix Double -> Matrix Double
        addOnesAndMultiply input weights =
          let (nbInput, _) = size input
              input' = konst 1 (nbInput, 1) ||| input
          in cmap h (input' <> weights)

        h = getActivationFunction nn

-- Train the Neural Network with Backpropagation algorithm, make N passes on the input
trainNTimes :: (NeuralNetwork, [Double]) -> TrainingSet -> Int -> LearningRateStrategy -> BackPropStrategy -> (NeuralNetwork, [Double])
trainNTimes (nn, c) _ 0 _ _ = (nn, c)
trainNTimes (nn, c) trainingSet nTimes (FixedRate alpha) backpropStrat =
  let newNN = trainOnce nn trainingSet alpha backpropStrat
      input = (4><2) (concat [[0,0],[0,1],[1,0],[1,1]]) :: Matrix Double
      target = (4><1) [0,1,1,0] :: Matrix Double
      output = runNN nn input
      newC  = sumElements $ cmap (^2) (output - target)
  -- Should shuffle the training set after each pass, to avoid cycling
  in {-trace ("#" ++ show nTimes) $ -} {-trace ("Weights:\n" ++ show (getWeights newNN) ++ "\n") $-} trainNTimes (newNN, newC:c) trainingSet (nTimes - 1) (FixedRate alpha) backpropStrat

trainOnce :: NeuralNetwork -> TrainingSet -> Double -> BackPropStrategy -> NeuralNetwork
trainOnce nn trainingSet alpha BatchGradientDescent =
  let zeroDeltas = initEmptyDeltas (getStructure nn)
      accDeltas  = foldl' (updateNetwork nn) zeroDeltas trainingSet
      !partialDerivatives = map (/ m) accDeltas :: [Matrix Double]
      currWeights = getWeights nn
      updatedWeights = zipWith updateWeights currWeights partialDerivatives :: [WeightMatrix]
  in nn {weights = updatedWeights}
   
      where initEmptyDeltas :: Structure -> [Matrix Double]
            initEmptyDeltas [_] = []
            initEmptyDeltas (l1:l2:xs) =
              let l1' = l1 + 1
              in (l1'><l2) (repeat 0) : initEmptyDeltas (l2:xs)

            m = fromIntegral . length $ trainingSet

            updateWeights :: WeightMatrix -> Matrix Double -> WeightMatrix
            updateWeights w deriv = w - scale alpha deriv

updateNetwork :: NeuralNetwork -> [Matrix Double] -> TrainingExample -> [Matrix Double]
updateNetwork nn deltas (input, target) =
  let vectorWeights = V.fromList (getWeights nn)
      -- First, compute all the unit's logits
      zs = V.scanl forwardPass input vectorWeights

      -- Second, compute error (delta) vectors for each layer
      ds = V.scanr backprop (cmap h (V.last zs) - target) $ V.zip (V.tail . V.init $ zs) (V.tail vectorWeights)

      -- Third, compute the Deltas
  in zipWith3 accumDeltas deltas (V.toList zs) (V.toList ds)

        where forwardPass :: Matrix Double -> WeightMatrix -> Matrix Double
              forwardPass lastZ w = let lastA = cmap h lastZ -- element-wise sigmoid
                                        lastA' = konst 1 (1,1) === lastA -- add biais
                                    in tr w <> lastA' -- compute next logit

              backprop :: (Matrix Double, WeightMatrix) -> Matrix Double -> Matrix Double
              backprop (z, w) d = let prod  = dropRows 1 (w <> d)
                                      deriv = cmap sigmoid' z
                                  in prod * deriv -- element-wise product here

              accumDeltas :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
              accumDeltas delta z d = let a = konst 1 (1,1) === cmap h z
                                      in delta + a <> tr d

              h = getActivationFunction nn

test :: IO ()
test = do
  nn <- mkNeuralNetwork sigmoid [2,2,1]
  let raw = [[0,0],[0,1],[1,0],[1,1]] :: [[Double]]
      rawSet = map (2><1) raw :: [Matrix Double]
      m = length raw
      n = length (head raw)
      input = (m><n) (concat raw) :: Matrix Double
  trace "Initial run:\n" $ putStrLn $ show $ runNN nn input
  let target = map (1><1) [[0],[1],[1],[0]] :: [Matrix Double]
      trainingSet = zip rawSet target
      output = runNN nn input
      initialCost = sumElements $ cmap (^2) (output - (4><1) [0,1,1,0])
  --putStrLn $ "training set: " ++ show trainingSet
      --newNN = trainOnce n trainingSet 0.1 BatchGradientDescent
  let (newNN, costs) = trainNTimes (nn, [initialCost]) trainingSet 1000 (FixedRate 0.5) BatchGradientDescent
  --trace ("test newNN:\n" ++ show (getWeights newNN)) $ return ()
  trace "Run after learning" $ putStrLn $ show $ runNN newNN input
  trace "Costs:\n" $ putStrLn $ show $ reverse costs
