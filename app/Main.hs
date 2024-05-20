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
import System.IO.Unsafe (unsafePerformIO)

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

import Data.List.Split (splitOn)
import qualified Data.List as List

data Temperature = Temperature {
  date :: String,
  daily_mean_temprature :: Float }
  deriving (Generic,Show)

-- instance FromRecord Temperature
-- instance ToRecord Temperature
instance FromNamedRecord Temperature where
    parseNamedRecord r = Temperature <$> r .: "date" <*> r .: "daily_mean_temprature"


model :: Linear -> Tensor -> Tensor
model state input = squeezeAll $ linear state input

groundTruth :: Tensor -> Tensor
groundTruth t = squeezeAll $ matmul t weight + bias
  where
    weight = asTensor ([42.0, 64.0, 96.0] :: [Float])
    bias = full' [1] (3.14 :: Float)

printParams :: Linear -> IO ()
printParams trained = do
  putStrLn $ "Parameters:\n" ++ (show $ toDependent $ trained.weight)
  putStrLn $ "Bias:\n" ++ (show $ toDependent $ trained.bias)

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

main :: IO ()
main = do
  -- ファイル読み込み
  train <- BL.readFile "/home/acf16406dh/hasktorch-projects/app/linearRegression/datas/train.csv"

  -- float型の気温のリストを作る
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  let train_tempature_list = case decodeByName train of
        Left error -> [] -- errorの時Left msgが返される
        Right (_, vector_tempature) -> convertToFloatLists vector_tempature -- 最初の要素:ヘッダー情報を無視
  --print train_tempature_list


  -- 7日間の気温のリストと8日目の気温の組のリスト
  let temperaturePairs = makeTemperaturePairsList train_tempature_list
  print temperaturePairs


  init <- sample $ LinearSpec {in_features = numFeatures, out_features = 1}
  randGen <- defaultRNG
  printParams init
  (trained, _) <- foldLoop (init, randGen) numIters $ \(state, randGen) i -> do
    let (input, randGen') = randn' [batchSize, numFeatures] randGen
        (y, y') = (groundTruth input, model state input)
        loss = mseLoss y y'
    when (i `mod` 100 == 0) $ do
      putStrLn $ "Iteration: " ++ show i ++ " | Loss: " ++ show loss
    (newParam, _) <- runStep state optimizer loss 5e-3
    pure (newParam, randGen')
  printParams trained
  pure ()
  where
    optimizer = GD
    defaultRNG = mkGenerator (Device CPU 0) 31415
    batchSize = 4
    numIters = 2000
    numFeatures = 3