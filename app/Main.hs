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
-- import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString as B
import qualified Data.ByteString.Char8 as C
-- from cassava
import Data.Csv.Incremental
import Data.Csv (FromRecord, ToRecord)

import Data.List.Split (splitOn)


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



data WeatherData = WeatherData
  { date :: !ByteString
  , daily_mean_temprature :: !Double
  } deriving(Show, Eq, Generic)

instance FromRecord WeatherData
instance ToRecord WeatherData


-- ([7日間のデータ], 8日目の気温)を作りたい

readFromFileToList :: FilePath -> IO [([Float], Float)]
readFromFileToList filename = do
  -- CSVファイルの読み込み
  -- B.readFile :: FilePath -> IO ByteString
  csvData <- B.readFile filename
  -- B.putStr csvData
  
  -- 行ごとに分割してリストに入れる
  let list_csv = C.lines csvData
  let linesWithoutHeader = tail list_csv
  print linesWithoutHeader
  let temperatures = map (read . last . splitOn ",") linesWithoutHeader

  


  -- parse datas
  -- まず気温だけのリストにする

  -- [([Float], Float)]への変換

  let dataList = [] -- あとでちゃんと代入する

  -- 気温データだけ取り出す
  --let linesWithoutHeader = tail (lines csvData)  -- ヘッダーを除く
  --put linesWithoutHeader
  
  return dataList


main :: IO ()
main = do
  dataList <- readFromFileToList "/home/acf16406dh/hasktorch-projects/app/linearRegression/datas/train.csv"


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