{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}


{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE DeriveGeneric     #-}
{-# LANGUAGE LambdaCase        #-}
{-# LANGUAGE OverloadedStrings #-}

module Main where

import Control.Monad (when)
import Torch
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

-- from base
import GHC.Generics
import System.IO
import System.Exit (exitFailure)

-- from bytestring
import Data.ByteString (ByteString, hGetSome, empty)
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C

-- from cassava
import Data.Csv
import Data.Text (Text)
import qualified Data.Vector as V
import Data.Vector as V hiding ((++), map, take, tail, filter, length, drop)
import qualified Data.List as List

data Temperature = Temperature {
  date :: String,
  daily_mean_temprature :: Float }
  deriving (Generic,Show)

-- instance FromRecord Temperature
-- instance ToRecord Temperature
instance FromNamedRecord Temperature where
    parseNamedRecord r = Temperature <$> r .: "date" <*> r .: "daily_mean_temprature"

-- 7日間の気温リストと翌日の気温をペアにする関数
makeTemperaturePairsList :: [Float] -> [([Float], Float)]
makeTemperaturePairsList temperatureList
  | length temperatureList < 8 = []
  | otherwise = (Prelude.take 7 temperatureList, temperatureList !! 7) : makeTemperaturePairsList (tail temperatureList)

-- Tempature型を受け取ったらdaily_mean_tempratureを返す
convertToTemprature :: Temperature -> Float
convertToTemprature = daily_mean_temprature

-- Vector Tempatureを受け取ったらfloatのリストを返す
-- toList :: Vector a -> [a]
convertToFloatLists :: (V.Vector Temperature) -> [Float]
convertToFloatLists vector_tempature =
  let tempature_list = V.toList vector_tempature
  in map convertToTemprature tempature_list

-- データをテンソルに変換
prepareData :: [([Float], Float)] -> [(Tensor, Tensor)]
prepareData dataPairs = map (\(inputs, output) -> (asTensor inputs, asTensor [output])) dataPairs

-- 出力を計算
model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

printParams :: Linear -> IO ()
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)



main :: IO ()
main = do
  -- ファイル読み込み
  trainingData <- BL.readFile "/home/acf16406dh/hasktorch-projects/app/linearRegression/datas/train.csv"

  -- float型の気温のリストを作る
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  let train_tempature_list = case decodeByName trainingData of
        Left error -> [] -- errorの時Left msgが返される
        Right (_, vector_tempature) -> convertToFloatLists vector_tempature -- 最初の要素:ヘッダー情報を無視

  -- 7日間の気温のリストと8日目の気温の組のリスト
  let temperaturePairs = makeTemperaturePairsList train_tempature_list

  -- 7日間の気温のリストと8日目の気温の組のリストをTensorの組のリストに変換
  let preparedData = prepareData temperaturePairs

  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1} -- モデルの初期化, 入力次元数と出力次元数を指定
  printParams init -- 初期化されたモデルの重みとバイアスを表示

  -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  -- foldLoop x count block = foldM block x [1 .. count]
  -- block関数を適用しながら初期値xを更新していく
  -- block関数は現在のモデル状態とindex iを引数に取って更新された値を返す

  (trained, losses) <- foldLoop (init, []) numIters $ \(state, losses) i -> do -- stateは現在のモデル状態, iは現在のイテレーション番号
    (trained, lossValue) <- foldLoop (init, 0) (length temperaturePairs) $ \(state, lossValue) j -> do
      let (inputData, outputData) = temperaturePairs !! (j `mod` length temperaturePairs) -- データポイントを取得, length temperaturePairs:2915
          input = asTensor inputData
          output = asTensor outputData
          (y, y') = (output, model state input)
          loss = mseLoss y y' -- 平均2乗誤差
          lossValue = asValue loss :: Float
      when (i `mod` 100 == 0) $ do
        putStrLn $ "Iteration: " ++ show j ++ " | Loss: " ++ show loss
      (newParam, _) <- runStep state optimizer loss 1e-6 -- パラメータを更新 学習率
      pure(newParam, lossValue)
    pure (trained, losses ++ [lossValue]) -- epochごとにlossを更新したい
  printParams trained
  drawLearningCurve "/home/acf16406dh/hasktorch-projects/app/linearRegression/curves/graph-avg_batch1.png" "Learning Curve" [("",losses)]
  pure ()
  where
    optimizer = GD -- 勾配降下法
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 2048 -- バッチサイズ, 一度に処理するデータのサンプル数
    numIters = 310 -- 何回ループ回すか
    numFeatures = 7 -- 入力の特徴数

    -- y = ax + b 
    -- a: weight, b: bias

    -- 学習率α「αが大きければ一度に大きく更新し、小さければ一度に少しずつ更新する」