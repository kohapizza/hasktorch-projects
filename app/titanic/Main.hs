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
  -- 訓練用データの読み込み
  treatedData <- treatData "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
  let passengerPairs = makePairsList treatedData -- ([他の情報], 生存)のリスト
  -- print $ take 5 passengerPairs -- OK
  -- print $ length passengerPairs -- 712

  -- データをシャッフル
  -- shuffleM :: MonadRandom m => [a] -> m [a]
  sffuledPassengerPairs <- shuffleM passengerPairs

  -- データをトレーニング用と評価用に分ける
  -- 20%(142)を検証用に, 80%(570)をトレーニング用に使う
  let (trainingData, validationData) = (take 570 sffuledPassengerPairs, drop 570 sffuledPassengerPairs)
  -- print $ take 5 trainingData -- OK
  -- 一例:
  -- [([1.0,0.0,22.0,0.0,0.0,151.55,1.0],1.0),([3.0,1.0,32.0,0.0,0.0,8.05,1.0],1.0),([3.0,1.0,41.0,0.0,0.0,7.125,1.0],0.0),([1.0,0.0,26.0,0.0,0.0,78.85,1.0],1.0),([3.0,1.0,22.0,0.0,0.0,7.7958,1.0],0.0)]
  -- print $ take 5 validationData -- OK

  -- テスト用データの読み込み
  treatedTestData <- treatTestData "/home/acf16406dh/hasktorch-projects/app/titanic/data/test.csv"
  print $ length treatedTestData -- 331 本当は 419

  -- 設定
  let epoch = 300::Int
      batchSize = 64::Int
      device = Device CUDA 0
      hypParams = MLPHypParams device 7 [(60,Sigmoid),(1,Sigmoid)] -- 入力層のノード数:7,隠れ層のノード層:60,出力層:1

  -- 初期モデル
  initModel <- sample hypParams

  -- mapAccumM :: (Monad m, Foldable t) => t a -> b -> (a -> b -> m (b,c)) -> m (b, [c])
  -- [1..iter] : 畳み込むリスト
  -- (initModel, GD) : アキュムレーターの初期値
  -- その後の2引数関数 : 各要素の対して適用する関数。epoch : リストの現在の要素。 (model, opt) : アキュムレータのタプル

  -- print $ length $ makeBatches trainingData batchSize -- 9
  -- print $ length $ makeBatches validationData batchSize -- 3

  -- バッチサイズ分のデータが何個取れるか
  let iterForTrain = (length $ makeBatches trainingData batchSize) -1
  let iterForValid = (length $ makeBatches validationData batchSize) -1


  -- 外側のループ: epoch処理, 内側のループ: バッチ処理
  ((trainedModel,_),losses) <- mapAccumM [1..epoch] (initModel,GD) $ \epoc (model,opt) -> do -- 各エポックでモデルを更新し、損失を出していく
    
    let trainLoss = sumTensors $ for (init (makeBatches trainingData batchSize) ) $ \batch -> -- 最後のバッチサイズ分無い要素は取り除く
                  
                  let loss = sumTensors $ for batch $ \(input, grandTruth) -> -- loss:各バッチでのloss
                        let y = asTensor'' device grandTruth -- 正解データ
                            y' = mlpLayer model $ asTensor'' device input -- 予測データ
                        in mseLoss y y' -- 誤差計算 mseLoss : Tensor
                        
                  -- fromIntegral :: (Integral a, Num b) => a -> b
                  -- in loss / fromIntegral batchSize -- バッチサイズで割る
                  -- バッチサイズで割るのではなく、length makeBatches trainingData batchSize で割るべき
                  -- in loss / length makeBatches trainingData batchSize
                  
                  in loss / fromIntegral batchSize
        trainLossValue = (asValue (trainLoss / fromIntegral iterForTrain))::Float -- trainLoss:1エポックでのバッチの合計loss
    -- epochが10の倍数ごとにLossを表示
    showLoss 10 epoc trainLossValue 

    

    -- モデルの更新
    u <- update model opt trainLoss 1e-3

    let validLoss = sumTensors $ for (init (makeBatches validationData batchSize) ) $ \batch ->
                  let loss = sumTensors $ for batch $ \(input,groundTruth) ->
                        let y = asTensor'' device groundTruth
                            y' = mlpLayer (fst u) $ asTensor'' device input
                        in mseLoss y y'
                  in loss / fromIntegral batchSize
        validLossValue = (asValue (validLoss / fromIntegral iterForValid))::Float  -- 消失テンソルをFloat値に変換
    return (u, (trainLossValue, validLossValue))
  
  -- モデルの保存
  -- saveParams trainedModel "/home/acf16406dh/hasktorch-projects/app/titanic/curves/model.pt"

  -- モデルの再利用
  -- model <- loadParams hypParams "/home/acf16406dh/hasktorch-projects/app/titanic/curves/model.pt"
  
  let (trainLosses, validLosses) = unzip losses   -- lossesを分解する
  drawLearningCurve "/home/acf16406dh/hasktorch-projects/app/titanic/curves/graph2.png" "Learning Curve" [("Training", reverse trainLosses), ("Validation", reverse validLosses)]
  -- print trainedModel
  where for = flip map