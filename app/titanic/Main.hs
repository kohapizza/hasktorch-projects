{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE OverloadedStrings #-}

import Data.List.Split (splitOn)
import qualified Data.ByteString.Lazy as BL
import System.IO
import Data.List.Utils (replace)
import Data.Csv as Csv
import qualified Data.Vector as V
import Data.List (sort)
import GHC.Generics (Generic)
import Data.Maybe (fromMaybe, isNothing)

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


-- リストからいらない列(index番目)を消す
deleteColumn :: Int -> [Float] -> [Float]
deleteColumn index row = take index row ++ drop (index + 1) row

-- リストから複数のいらない列を消す
-- foldl: リストを畳み込む
-- 「左畳み込み」と呼ばれ、リストの要素を左から右へと順番に処理する
deleteColumns :: [Int] -> [Float] -> [Float]
deleteColumns idxs row = foldl (flip deleteColumn) row (reverse (sort idxs))

-- 全ての行から複数の列を削除
deleteAllColumns :: [[Float]] -> [Int] -> [[Float]]
deleteAllColumns parsedCsvData columnsToDelete = map (deleteColumns columnsToDelete) parsedCsvData

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
  print $ take 10 treatedData