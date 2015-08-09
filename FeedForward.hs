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
import qualified Data.Vector as V (singleton, scanl', scanr', fromList, toList, zip, last, head, tail, init, zipWith3, cons)
import           Debug.Trace (trace)
import           Numeric.LinearAlgebra.HMatrix hiding (Vector)
import           System.Random (randomRs, newStdGen)
import           System.Random.MWC

import Graphics.Rendering.Plot

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
v-}
mkNeuralNetwork :: ActivationFunction -> Structure -> IO NeuralNetwork
mkNeuralNetwork _ xs | length xs < 2 = error "A Neural Network must have at least one input layer and an ouput layer, so your structure must contain at least 2 numbers"
                     | any (<= 0) xs = error "You can't have zero or a negative number of units in a layer"
mkNeuralNetwork h xs = do
  !initRandomWeights <- genInitWeights (zip xs (tail xs))

  return $ NeuralNetwork
    { structure          = xs
    , weights            = initRandomWeights
    , activationFunction = h
    }

    where genInitWeights = mapM $ \(n1,n2) -> do
            --let bound = 0.5 * sqrt (6 / (fromIntegral (n1+n2))) :: Double -- bound value taken from Andrew Ng's class
            let bound = 1.0 / sqrt (fromIntegral n1)
                n1' = n1 + 1
            vs <- withSystemRandom . asGenST $ \gen -> uniformVector gen (n1'*n2)
            return $ cmap (\w -> (w * 2 * bound) - bound) $ (n1'><n2) $ V.toList vs -- zero-mean the weights

-- Run the Neural Network on the input matrix to get output matrix (automatically add biais neurons with value 1)
runNN :: NeuralNetwork -> Matrix Double -> Matrix Double
runNN nn input =
  let ws = weights nn
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
      newC  = 0.5 * (sumElements $ cmap (^2) (output - target))
  -- Should shuffle the training set after each pass, to avoid cycling
  in trainNTimes (newNN, newC:c) trainingSet (nTimes - 1) (FixedRate alpha) backpropStrat

trainOnce :: NeuralNetwork -> TrainingSet -> Double -> BackPropStrategy -> NeuralNetwork
trainOnce nn trainingSet alpha BatchGradientDescent =
  let !zeroDeltas = initEmptyDeltas (getStructure nn)
      !accDeltas  = foldl' (updateNetwork nn) zeroDeltas trainingSet
      -- !partialDerivatives = map (/ m) accDeltas :: [Matrix Double]
      !rescaledDeltas = map (/ m) accDeltas :: [Matrix Double]
      currWeights = getWeights nn
      !partialDerivatives = zipWith (regularize 0) rescaledDeltas currWeights
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

            regularize :: Double -> Matrix Double -> WeightMatrix -> Matrix Double
            regularize lambda rDelta weight = let rDelta' = dropRows 1 rDelta
                                                  weight' = dropRows 1 weight
                                                  regu    = rDelta' + (scale lambda weight')
                                              in takeRows 1 rDelta === regu

updateNetwork :: NeuralNetwork -> [Matrix Double] -> TrainingExample -> [Matrix Double]
updateNetwork nn deltas (input, target) =
  let vectorWeights = V.fromList (getWeights nn)

      {- First, compute all the unit's logits
         Because of forwardPass's first action is to apply activation function,
         we can't simply call 'scanl forwardPass input vectorWeights', so we manually
         compute the first pass.
         But the input won't appear in the return vector from scanl so we prepend it.
      -}
      input' = konst 1 (1,1) === input
      w1     = (V.head vectorWeights)
      zs = input `V.cons` V.scanl' forwardPass (tr w1 <> input') (V.tail vectorWeights)

      -- Second, compute error (delta) vectors for each layer
      ds = V.scanr' backprop (cmap h (V.last zs) - target) $ V.zip (V.tail . V.init $ zs) (V.tail vectorWeights)

      -- Third, compute the Deltas
  in zipWith3 accumDeltas deltas (V.toList zs) (V.toList ds)

        where forwardPass :: Matrix Double -> WeightMatrix -> Matrix Double
              forwardPass lastZ w = let lastA = cmap h lastZ -- element-wise sigmoid
                                        lastA' = konst 1 (1,1) === lastA -- add biais
                                    in tr w <> lastA' -- compute next logit

              backprop :: (Matrix Double, WeightMatrix) -> Matrix Double -> Matrix Double
              backprop (z, w) d = let prod  = (dropRows 1 w) <> d
                                      deriv = cmap sigmoid' z
                                  in prod * deriv -- element-wise product here

              accumDeltas :: Matrix Double -> Matrix Double -> Matrix Double -> Matrix Double
              accumDeltas delta z d = let a = konst 1 (1,1) === cmap h z
                                      in delta + (a <> tr d)

              h = getActivationFunction nn

-- usage: test "plot.png"
test :: FilePath -> IO ()
test fp = do
  nn <- mkNeuralNetwork sigmoid [2,2,1]
  let raw = [[0,0],[0,1],[1,0],[1,1]] :: [[Double]]
      rawSet = map (2><1) raw :: [Matrix Double]
      m = length raw
      n = length (head raw)
      input = (m><n) (concat raw) :: Matrix Double
      nbPasses = 1000
      alpha = 3.5
  putStrLn $ "Initial Weights:\n"
  mapM_ (putStrLn . show) (getWeights nn)
  putStrLn "Initial run:\n"
  putStrLn $ show $ runNN nn input

  let target = map (1><1) [[0],[1],[1],[0]] :: [Matrix Double]
      trainingSet = zip rawSet target
      output = runNN nn input
  let (newNN, costs) = trainNTimes (nn, []) trainingSet nbPasses (FixedRate alpha) BatchGradientDescent
  --putStrLn "Final Weights:\n"
  --mapM_ (putStrLn . show) (getWeights newNN)
  putStrLn "Final run:\n"
  putStrLn $ show $ runNN newNN input
--  putStrLn "\nCosts:\n"
--  putStrLn $ show $ head $ costs
--  putStrLn $ show $ reverse costs

  -- Displaying the cost in the file
  let ts = vector [1.. fromIntegral nbPasses]
  let graph = do
        plot (ts, [linepoint (vector (reverse costs)) (1.0::LineWidth, blue) (Cross, blue)])

        title "Cost Function"
        subtitle $ "alpha = " ++ show alpha ++ "\n" ++ show nbPasses ++ " passes"

        xlabel "# Pass"
        ylabel "Cost"
        grid True

  writeFigure PNG fp (1024, 768) graph
