{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Data.List.Split (splitOn)
import qualified Data.ByteString.Lazy as BL
import System.IO
import Data.List.Utils (replace)
import Data.Csv as Csv
import qualified Data.Vector as V
import Data.List (sort)
import GHC.Generics (Generic)
import Data.Maybe (fromMaybe, isNothing)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)
import Prelude hiding (tanh) 
import Control.Monad (forM_, foldM)
import Torch.Tensor (asValue)
import Torch.Functional (mseLoss)
import Torch.Device (Device(..), DeviceType(..))
import Torch.NN (sample)
import Torch.Train (update, showLoss, sumTensors, saveParams, loadParams)
import Torch.Control (mapAccumM)
import Torch.Optim (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer)
import ML.Exp.Chart (drawLearningCurve)
import System.Random.Shuffle

-- train.csv用のdata構造
data Passenger = Passenger {
  survived :: Maybe Int,
  pclass :: Maybe Int,
  sex :: Maybe String,
  age :: Maybe Float,
  sibSp :: Maybe Int,
  parch :: Maybe Int,
  fare :: Maybe Float,
  embarked :: Maybe String
} deriving (Generic, Show)

instance Csv.FromNamedRecord Passenger where
    parseNamedRecord m = Passenger <$> m .: "Survived"
                                   <*> m .: "Pclass"
                                   <*> m .: "Sex"
                                   <*> m .: "Age"
                                   <*> m .: "SibSp"
                                   <*> m .: "Parch"
                                   <*> m .: "Fare"
                                   <*> m .: "Embarked"

-- 性別をfloatに
sexToFloat :: Maybe String -> Maybe Float
sexToFloat (Just "female") = Just 0.0
sexToFloat (Just "male") = Just 1.0
sexToFloat _ = Nothing

-- 出港地をfloatに
embarkedToFloat :: Maybe String -> Maybe Float
embarkedToFloat (Just "Q") = Just 0.0
embarkedToFloat (Just "S") = Just 1.0
embarkedToFloat (Just "C") = Just 2.0
embarkedToFloat _ = Nothing

convertToFloatLists :: V.Vector Passenger -> [[Float]]
convertToFloatLists vectorData = 
  let passengerList = V.toList vectorData
  in map convertPassenger (filter isCompleteData passengerList)

isCompleteData :: Passenger -> Bool
isCompleteData p =
  all (not . isNothing)
    [ survived p, pclass p, sibSp p, parch p ] &&
  all (not . isNothing)
    [ sexToFloat (sex p), embarkedToFloat (embarked p) ] && 
  all (not . isNothing)
    [ fare p, age p ]

convertPassenger :: Passenger -> [Float]
convertPassenger (Passenger mSurvived mPclass mSex mAge mSibSp mParch mFare mEmbarked) =
  [ fromMaybe 0 (fmap fromIntegral mSurvived),
    fromMaybe 0 (fmap fromIntegral mPclass),
    fromMaybe 0 (sexToFloat mSex),
    fromMaybe 0 mAge,
    fromMaybe 0 (fmap fromIntegral mSibSp),
    fromMaybe 0 (fmap fromIntegral mParch),
    fromMaybe 0 mFare,
    fromMaybe 0 (embarkedToFloat mEmbarked)
  ]

makePair :: [Float] -> ([Float], Float)
makePair passenger = (tail passenger, passenger !! 0)

makePairsList :: [[Float]] -> [([Float], Float)]
makePairsList passengerList = map makePair passengerList

makeBatches :: [a] -> Int -> [[a]]
makeBatches [] _ = []
makeBatches xs n = take n xs : makeBatches (drop n xs) n

-- 訓練データのフォーマットを整える
treatData :: FilePath -> IO [[Float]]
treatData filePath = do
  csvData <- BL.readFile filePath -- ファイル読み込み
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  case decodeByName csvData of
        Left error -> do
          putStrLn $ "Error parsing CSV: " ++ error
          return []
        Right (_, v) -> do
          let floatLists = convertToFloatLists v
          return floatLists

-- 指標の計算
data Metrics = Metrics {
    accuracy :: Float,
    precision :: Float,
    recall :: Float
} deriving Show

calculateMetrics :: [Float] -> [Float] -> Metrics
calculateMetrics predictions actuals = 
    let
        tp = fromIntegral $ length $ filter (\(p, a) -> p == 1 && a == 1) $ zip predictions actuals -- 予測と値のペアのリストにする
        tn = fromIntegral $ length $ filter (\(p, a) -> p == 0 && a == 0) $ zip predictions actuals
        fp = fromIntegral $ length $ filter (\(p, a) -> p == 1 && a == 0) $ zip predictions actuals
        fn = fromIntegral $ length $ filter (\(p, a) -> p == 0 && a == 1) $ zip predictions actuals
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        -- 分母が0になるときの実装！！！！
    in
        Metrics accuracy precision recall

main :: IO ()
main = do
    -- 訓練用データの読み込みと前処理
    treatedData <- treatData "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
    let passengerPairs = makePairsList treatedData
    
    -- データのシャッフル
    sffuledPassengerPairs <- shuffleM passengerPairs
    
    -- トレーニング用と評価用データに分ける
    let (trainingData, validationData) = (take 570 sffuledPassengerPairs, drop 570 sffuledPassengerPairs)
    
    -- 評価用データの処理
    let validInputs = map fst validationData
    let validActuals = map snd validationData

    -- 設定
    let epoch = 300::Int
        batchSize = 64::Int
        device = Device CUDA 0
        hypParams = MLPHypParams device 7 [(60,Sigmoid),(1,Sigmoid)] -- 入力層のノード数:7,隠れ層のノード層:60,出力層:1
    
    -- モデルのロード
    model <- loadParams hypParams "/home/acf16406dh/hasktorch-projects/app/titanic/curves/model_batch64.pt"
    
    -- 評価データに対する予測を行う
    let predictions = map (\input -> 
                          let y' = mlpLayer model $ asTensor'' device input
                              yFloat = asValue y'::Float
                          in if (yFloat > 0.5) then 1.0 else 0.0
                         ) validInputs
    
    -- 指標の計算
    let metrics = calculateMetrics predictions validActuals
    print metrics
