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

-- from random
import System.Random.Shuffle(shuffleM)
import System.Random (newStdGen)

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

-- -- 0~Int-1のランダムな整数のリストを生成する関数
-- randomList :: Int -> IO [Int]
-- randomList n = do
--   shuffleM [0..n-1] -- リストをシャッフル

-- モデル評価関数
validModel :: Linear -> [([Float], Float)] -> IO Float
validModel state validData = do
  let (inputData, outputData) = Prelude.unzip validData
      input = asTensor inputData
      output = asTensor outputData
      (y, y') = (output, model state input)
      loss = mseLoss y y' -- 平均2乗誤差
  return $ asValue loss

  

          
main :: IO ()
main = do
  -- for training
  -- ファイル読み込み
  trainingData <- BL.readFile "/home/acf16406dh/hasktorch-projects/app/linearRegression/datas/train.csv"

  -- float型の気温のリストを作る
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  let train_tempature_list = case decodeByName trainingData of
        Left error -> [] -- errorの時Left msgが返される
        Right (_, vector_tempature) -> convertToFloatLists vector_tempature -- 最初の要素:ヘッダー情報を無視

  -- 7日間の気温のリストと8日目の気温の組のリスト
  let temperaturePairs = makeTemperaturePairsList train_tempature_list

  -- for Validation 過学習の確認のため使うデータ
  -- ファイル読み込み
  validaionData <- BL.readFile "/home/acf16406dh/hasktorch-projects/app/linearRegression/datas/valid.csv"

  -- float型の気温のリストを作る
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  let valid_tempature_list = case decodeByName validaionData of
        Left error -> [] -- errorの時Left msgが返される
        Right (_, vector_tempature) -> convertToFloatLists vector_tempature -- 最初の要素:ヘッダー情報を無視

  -- 7日間の気温のリストと8日目の気温の組のリスト
  let validTemperaturePairs = makeTemperaturePairsList valid_tempature_list



  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1} -- モデルの初期化, 入力次元数と出力次元数を指定
  printParams init -- 初期化されたモデルの重みとバイアスを表示

  initRandomData <- shuffleM temperaturePairs -- train 1回目の学習データ用
  initValidRandomData <- shuffleM validTemperaturePairs -- valid 1回目の学習データ用

  -- foldLoop :: a -> Int -> (a -> Int -> IO a) -> IO a
  -- foldLoop x count block = foldM block x [1 .. count]
  -- block関数を適用しながら初期値xを更新していく
  -- block関数は現在のモデル状態とindex iを引数に取って更新された値を返す

  -- 内側のループ: epoch処理, 外側のループ: バッチ処理
  (trained, trainLosses, validlossess) <- foldLoop (init, [], []) numEpoch $ \(state, trainLosses, validlossess) i -> do -- stateは現在のモデル状態, iは現在のイテレーション番号
    -- randomedListからbatchsize分をtake
    -- そこに書いてある整数をtemperaturePairsから取り出す
    -- for train
    (trained', lossValue, randomData) <- foldLoop (state, 0, initRandomData) ((length temperaturePairs) `Prelude.div` batchSize) $ \(state', lossValue, randomData) j -> do
      let index = (j-1)*batchSize -- サブセットj個目か
          inputDataList = Prelude.take batchSize (drop index randomData) -- バッチサイズ分だけ取ってくる
          (inputData, outputData) = Prelude.unzip inputDataList
          input = asTensor inputData
          output = asTensor outputData
          (y, y') = (output, model state' input)
          loss = mseLoss y y' -- 平均2乗誤差
      when (j `mod` 10 == 0) $ do
        putStrLn $ "epoch : " ++ show i ++ " | Iteration: " ++ show j ++ " | Loss: " ++ show loss ++ " | losses : " ++ show trainLosses ++ " | lossValue : " ++ show lossValue 
      (newParam, _) <- runStep state' optimizer loss 1e-6 -- パラメータを更新 学習率
      pure(newParam, asValue loss, randomData)

    initRandomData <- shuffleM temperaturePairs -- train 1回目の学習データ用

    validlossValue <- validModel trained' validTemperaturePairs
    
    pure (trained', trainLosses ++ [lossValue], validlossess ++ [validlossValue]) -- epochごとにlossを更新したい


  printParams trained


  drawLearningCurve "/home/acf16406dh/hasktorch-projects/app/linearRegression/curves/graph-avg_batch128.png" "Learning Curve" [("Training",trainLosses),("Validation",validlossess)] 
  pure ()
  where
    optimizer = GD -- 勾配降下法
    -- defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 128 -- バッチサイズ, 一度に処理するデータのサンプル数
    numEpoch = 300 -- エポック数
    numFeatures = 7 -- 入力の特徴数

    -- y = ax + b 
    -- a: weight, b: bias
    
    -- 学習率α「αが大きければ一度に大きく更新し、小さければ一度に少しずつ更新する」