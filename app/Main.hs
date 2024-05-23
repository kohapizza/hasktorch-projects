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
  randGen <- defaultRNG -- 乱数生成器の初期化
  printParams init -- 初期化されたモデルの重みとバイアスを表示
  (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do -- stateは現在のモデル状態, randGenは現在の乱数生成器, iは現在のイテレーション番号
    let (inputData, outputData) = temperaturePairs !! (i `mod` length temperaturePairs) -- データポイントを取得
        (_ , randGen') = randn' [batchSize, numFeatures] randGen
        input = asTensor inputData
        output = asTensor outputData
        (y, y') = (output, model state input)
        loss = mseLoss y y' -- 平均2乗誤差
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newParam, _) <- runStep state optimizer loss 1e-7 -- パラメータを更新 学習率
    pure (newParam, randGen') -- 乱数生成期を更新
  printParams trained
  pure ()
  
  where
    optimizer = GD -- 勾配降下法
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 800 -- バッチサイズ, 一度に処理するデータのサンプル数
    numIters = 2000 -- 何回ループ回すか
    numFeatures = 7 -- 入力の特徴数

    -- y = ax + b 
    -- a: weight, b: bias

    -- 学習率α「αが大きければ一度に大きく更新し、小さければ一度に少しずつ更新する」