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
import Control.Monad (forM_)        --base
--import Data.List (cycle)          --base
--hasktorch
import Torch.Tensor       (asValue)
import Torch.Functional   (mseLoss)
import Torch.Device       (Device(..),DeviceType(..))
import Torch.NN           (sample)
import Torch.Train        (update,showLoss,sumTensors,saveParams,loadParams)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools
import System.Random.Shuffle

import Control.Monad (foldM) 

-- train.csv用のdata構造
-- dataが壊れている時はMaybeを使うといい
data Passenger = Passenger{
  survived :: Maybe Int,
  pclass :: Maybe Int,
  sex :: Maybe String,
  age :: Maybe Float,
  sibSp :: Maybe Int,
  parch :: Maybe Int,
  fare :: Maybe Float,
  embarked :: Maybe String
} deriving (Generic, Show)

-- test.csv用のdata構造
data PassengerForTest = PassengerForTest{
  pclassTest :: Maybe Int,
  sexTest :: Maybe String,
  ageTest :: Maybe Float,
  sibSpTest :: Maybe Int,
  parchTest :: Maybe Int,
  fareTest :: Maybe Float,
  embarkedTest :: Maybe String
} deriving (Generic, Show)

-- 訓練CSVデータからPassenger型のデータをデコードするためのインスタンスを定義
instance Csv.FromNamedRecord Passenger where
    parseNamedRecord m = Passenger <$> m .: "Survived"
                                   <*> m .: "Pclass"
                                   <*> m .: "Sex"
                                   <*> m .: "Age"
                                   <*> m .: "SibSp"
                                   <*> m .: "Parch"
                                   <*> m .: "Fare"
                                   <*> m .: "Embarked"

-- テストCSVデータからPassengerForTest型のデータをデコードするためのインスタンスを定義
instance Csv.FromNamedRecord PassengerForTest where
    parseNamedRecord m = PassengerForTest <$> m .: "Pclass"
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

-- Passenger型をリストに変換
convertToFloatLists :: V.Vector Passenger -> [[Float]]
convertToFloatLists vectorData = 
  let passengerList = V.toList vectorData -- Passengerのリストに
  in map convertPassenger (filter isCompleteData passengerList)
  

-- PassengerForTest型をリストに変換
convertToFloatListsForTest :: V.Vector PassengerForTest -> [[Float]]
convertToFloatListsForTest vectorData = 
  let passengerList = V.toList vectorData -- PassengerForTestのリストに
  in map convertPassengerForTest (filter isCompleteDataForTest passengerList)

-- 完全なデータ行かどうかをチェック
isCompleteData :: Passenger -> Bool
isCompleteData p =
  all (not . isNothing) -- allは型一緒じゃないと使えない
    [ survived p
    , pclass p
    , sibSp p
    , parch p
    ] &&
  all (not . isNothing)
    [ sexToFloat (sex p)
    , embarkedToFloat (embarked p)
    ] && 
  all (not . isNothing)
    [ fare p
    , age p
    ]

-- テストデータにおいて完全なデータ行かどうかをチェック
-- ここでデータが不完全なものは別の値で補完するようにする！！！！！！！！！！
isCompleteDataForTest :: PassengerForTest -> Bool
isCompleteDataForTest p =
  all (not . isNothing) -- allは型一緒じゃないと使えない
    [ pclassTest p
    , sibSpTest p
    , parchTest p
    ] &&
  all (not . isNothing)
    [ sexToFloat (sexTest p)
    , embarkedToFloat (embarkedTest p)
    ] && 
  all (not . isNothing)
    [ fareTest p
    , ageTest p
    ]

-- PassengerをFloatのリストに変換
convertPassenger :: Passenger -> [Float]
convertPassenger (Passenger mSurvived mPclass mSex mAge mSibSp mParch mFare mEmbarked) =
  [ fromMaybe 0 (fmap fromIntegral mSurvived)
  , fromMaybe 0 (fmap fromIntegral mPclass)
  , fromMaybe 0 (sexToFloat mSex)
  , fromMaybe 0 mAge
  , fromMaybe 0 (fmap fromIntegral mSibSp)
  , fromMaybe 0 (fmap fromIntegral mParch)
  , fromMaybe 0 mFare
  , fromMaybe 0 (embarkedToFloat mEmbarked)
  ]

-- PassengerForTestをFloatのリストに変換
convertPassengerForTest :: PassengerForTest -> [Float]
convertPassengerForTest (PassengerForTest mPclass mSex mAge mSibSp mParch mFare mEmbarked) =
  [ fromMaybe 0 (fmap fromIntegral mPclass)
  , fromMaybe 0 (sexToFloat mSex)
  , fromMaybe 0 mAge
  , fromMaybe 0 (fmap fromIntegral mSibSp)
  , fromMaybe 0 (fmap fromIntegral mParch)
  , fromMaybe 0 mFare
  , fromMaybe 0 (embarkedToFloat mEmbarked)
  ]

-- 生存とそれ以外の情報のペアにする関数
makePair :: [Float] -> ([Float], Float)
makePair passenger = (tail passenger, passenger !! 0)

-- 生存とそれ以外の情報のペアのリストにする関数
makePairsList :: [[Float]] -> [([Float], Float)]
makePairsList passengerList = map makePair passengerList

-- リストを指定されたバッチサイズに従って分割
-- ex: makeBatches [1,2,3,4,5,6,7,8,9,10] 3 => [[1,2,3],[4,5,6],[7,8,9],[10]]
makeBatches :: [a] -> Int -> [[a]]
makeBatches [] _ = []
makeBatches xs n = take n xs : makeBatches (drop n xs) n

-- テストデータのフォーマットを整える
treatTestData :: FilePath -> IO [[Float]]
treatTestData filePath = do
  csvData <- BL.readFile filePath -- ファイル読み込み
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  case decodeByName csvData of
        Left error -> do
          putStrLn $ "Error parsing CSV: " ++ error
          return []
        Right (_, v) -> do
          let floatLists = convertToFloatListsForTest v
          return floatLists


main :: IO ()
main = do
  -- テスト用データの読み込み
  treatedTestData <- treatTestData "/home/acf16406dh/hasktorch-projects/app/titanic/data/test.csv"
  print $ length treatedTestData -- 331 本当は 419

  -- モデルの再利用
  model <- loadParams hypParams "/home/acf16406dh/hasktorch-projects/app/titanic/curves/model.pt"
  
  let (trainLosses, validLosses) = unzip losses   -- lossesを分解する
  drawLearningCurve "/home/acf16406dh/hasktorch-projects/app/titanic/curves/graph_batch64.png" "Learning Curve" [("Training", reverse trainLosses), ("Validation", reverse validLosses)]
  -- print trainedModel
  where for = flip map