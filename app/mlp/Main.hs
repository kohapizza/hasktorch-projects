{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Control.Monad (when)
import Data.List (foldl', intersperse, scanl')
import GHC.Generics
import Torch

-- train.csvを扱いやすいデータにする
treatCSV :: filePath -> []

--------------------------------------------------------------------------------
-- MLP Multi Layer Perceptron：多層パーセプトロン
--------------------------------------------------------------------------------

data MLPSpec = MLPSpec -- MLPの仕様を表すデータ型
  { feature_counts :: [Int], -- 隠れ層のニューロン数のリスト
    nonlinearitySpec :: Tensor -> Tensor -- 非線形関数
  }

data MLP = MLP -- 実際のMLPモデルを表すデータ型
  -- data Linear = Linear {weight :: Parameter, bias :: Parameter} deriving (Show, Generic)
  { layers :: [Linear],
    nonlinearity :: Tensor -> Tensor
  }
  deriving (Generic, Parameterized)

instance Randomizable MLPSpec MLP where
  sample MLPSpec {..} = do
    let layer_sizes = mkLayerSizes feature_counts
    linears <- mapM sample $ map (uncurry LinearSpec) layer_sizes
    return $ MLP {layers = linears, nonlinearity = nonlinearitySpec}
    where
      mkLayerSizes (a : (b : t)) =
        scanl shift (a, b) t
        where
          shift (a, b) c = (b, c)

mlp :: MLP -> Tensor -> Tensor
mlp MLP {..} input = foldl' revApply input $ intersperse nonlinearity $ map linear layers
  where
    revApply x f = f x

--------------------------------------------------------------------------------
-- Training code
--------------------------------------------------------------------------------

batchSize = 2

numIters = 2000

model :: MLP -> Tensor -> Tensor
model params t = mlp params t

main :: IO ()
main = do
-- 初期モデルの作成
  init <-
-- sample :: spec -> IO f
    sample $
      MLPSpec
        { feature_counts = [2, 2, 1], -- 2つの入力, 2つの中間層のニューロン, 1つの出力を持つ
          nonlinearitySpec = Torch.tanh
        }
  -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  -- foldLoop x count block = foldM block x [1 .. count]
  -- block関数を適用しながら初期値xを更新していく
  -- block関数は現在のモデル状態とindex iを引数に取って更新された値を返す
  trained <- foldLoop init numIters $ \state i -> do
    -- ランダムなバッチデータを作成して右側の関数に渡す
    -- gt :: Tensor -> Tensor -> Tensor
    -- 各要素が0.5より大きいか調べて, Trueなら1, Falseなら0を返す
    -- toDType FloatでFloat型のTensorにする
    input <- randIO' [batchSize, 2] >>= return . (toDType Float) . (gt 0.5)
    -- squeezeAll :: Tensor -> Tensor
    -- squeezeAllでサイズが1の次元をすべて取り除く
    let (y, y') = (tensorXOR input, squeezeAll $ model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newState, _) <- runStep state optimizer loss 1e-1
    return newState
  putStrLn "Final Model:"
  putStrLn $ "0, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 0 :: Float]))
  putStrLn $ "0, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [0, 1 :: Float]))
  putStrLn $ "1, 0 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 0 :: Float]))
  putStrLn $ "1, 1 => " ++ (show $ squeezeAll $ model trained (asTensor [1, 1 :: Float]))
  return ()
  where
    optimizer = GD  -- 勾配降下法（Gradient Descent）
    tensorXOR :: Tensor -> Tensor -- xor関数を実装した補助関数
    tensorXOR t = (1 - (1 - a) * (1 - b)) * (1 - (a * b))
      where
        a = select 1 0 t
        b = select 1 1 t
