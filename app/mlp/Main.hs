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
import Torch.Train        (update,showLoss,sumTensors)
import Torch.Control      (mapAccumM)
import Torch.Optim        (GD(..))
import Torch.Tensor.TensorFactories (asTensor'')
import Torch.Layer.MLP    (MLPHypParams(..),ActName(..),mlpLayer)
import ML.Exp.Chart   (drawLearningCurve) --nlp-tools

-- passengerId： 乗客者ID  -- 消す
-- survived：生存状況（0＝死亡、1＝生存）
-- pclass： 旅客クラス（1＝1等、2＝2等、3＝3等）。裕福さの目安となる
-- name： 乗客の名前 -- 消す
-- sex： 性別（male＝男性、female＝女性）
-- age： 年齢。一部の乳児は小数値
-- sibsp： タイタニック号に同乗している兄弟（Siblings）や配偶者（Spouses）の数
-- parch： タイタニック号に同乗している親（Parents）や子供（Children）の数
-- ticket： チケット番号  -- 消す
-- fare： 旅客運賃
-- cabin： 客室番号  -- 消す
-- embarked： 出港地（C＝Cherbourg：シェルブール、Q＝Queenstown：クイーンズタウン、S＝Southampton：サウサンプトン）

-- data構造
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

-- CSVデータからPassenger型のデータをデコードするためのインスタンスを定義
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

-- Passenger型をリストに変換
convertToFloatLists :: V.Vector Passenger -> [[Float]]
convertToFloatLists vectorData = 
  let passengerList = V.toList vectorData -- Passengerのリストに
  in map convertPassenger (filter isComplete passengerList)

-- 完全なデータ行かどうかをチェック
isComplete :: Passenger -> Bool
isComplete p =
  all (not . isNothing)
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

-- 生存とそれ以外の情報のペアにする関数
makePair :: [Float] -> ([Float], Float)
makePair passenger = (tail passenger, passenger !! 0)


-- 生存とそれ以外の情報のペアのリストにする関数
makePairsList :: [[Float]] -> [([Float], Float)]
makePairsList passengerList = map makePair passengerList

-- リストを指定されたバッチサイズに従って分割
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
          -- let columnsToDelete = [0, 3, 8, 10] -- passengerID, name, ticket, cabinの削除
          let floatLists = convertToFloatLists v
          -- return $ deleteAllColumns floatLists columnsToDelete -- いらない列消す
          return floatLists


main :: IO ()
main = do
  treatedData <- treatData "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
  let passengerPairs = makePairsList treatedData -- ([他の情報], 生存)のリスト
  print $ take 5 passengerPairs
  print $ length passengerPairs -- 712

  -- データをトレーニング用と評価用に分ける
  -- 20%(142)を検証用に, 80%(570)をトレーニング用に使う
  let (trainingData, validationData) = (take 570 passengerPairs, drop 570 passengerPairs)
  print $ take 5 trainingData -- OK
  print $ take 5 validationData -- OK

  -- 設定
  let iter = 300::Int
      batchSize = 64::Int
      device = Device CUDA 0
      hypParams = MLPHypParams device 7 [(10,Sigmoid),(1,Sigmoid)]

  -- 初期モデル
  initModel <- sample hypParams

  ((trainedModel,_),losses) <- mapAccumM [1..iter] (initModel,GD) $ \epoc (model,opt) -> do -- 各エポックでモデルを更新し、損失を蓄積。
    let trainLoss = sumTensors $ for (makeBatches trainingData batchSize) $ \batch ->
                  let loss = sumTensors $ for batch $ \(input, grandTruth) ->
                        let y = asTensor'' device grandTruth
                            y' = mlpLayer model $ asTensor'' device input
                        in mseLoss y y' -- 誤差計算
                  in loss / fromIntegral batchSize
        trainLossValue = (asValue trainLoss)::Float

    let validLoss = sumTensors $ for (makeBatches validationData batchSize) $ \batch ->
                  let loss = sumTensors $ for batch $ \(input,groundTruth) ->
                        let y = asTensor'' device groundTruth
                            y' = mlpLayer model $ asTensor'' device input
                        in mseLoss y y'  -- 平均二乗誤差を計算
                  in loss / fromIntegral batchSize
        validLossValue = (asValue validLoss)::Float  -- 消失テンソルをFloat値に変換
    showLoss 10 epoc trainLossValue 
    u <- update model opt trainLoss 1e-3
    return (u, (trainLossValue, validLossValue))
  
  let (trainLosses, validLosses) = unzip losses   -- lossesを分解する
  drawLearningCurve "/home/acf16406dh/hasktorch-projects/app/mlp/curves/graph.png" "Learning Curve" [("Training", reverse trainLosses), ("Validation", reverse validLosses)]
  -- print trainedModel
  where for = flip map