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
data Passenger = Passenger{
  passengerID :: Int,
  survived :: Maybe Int,
  pclass :: Maybe Int,
  name :: String, 
  sex :: Maybe String,
  age :: Maybe Int,
  sibSp :: Maybe Int,
  parch :: Maybe Int,
  ticket :: String,
  fare :: Maybe Float,
  cabin :: String,
  embarked :: Maybe String
} deriving (Generic, Show)

-- CSVデータからPassenger型のデータをデコードするためのインスタンスを定義
instance Csv.FromNamedRecord Passenger where
    parseNamedRecord m = Passenger <$> m .: "PassengerId"
                                   <*> m .: "Survived"
                                   <*> m .: "Pclass"
                                   <*> m .: "Name"
                                   <*> m .: "Sex"
                                   <*> m .: "Age"
                                   <*> m .: "SibSp"
                                   <*> m .: "Parch"
                                   <*> m .: "Ticket"
                                   <*> m .: "Fare"
                                   <*> m .: "Cabin"
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

-- 後で
replaceRow :: [String] -> [String] -> [String] -> [String]
replaceRow oldThings newThings row = foldl (\r (old, new) -> map (replace old new) r) row (zip oldThings newThings)

-- 後で
-- 全ての行の'oldThings'を'newThings'に置き換える
-- replace :: Eq a => [a] -> [a] -> [a] -> [a]
replaceAll :: [String] -> [String] -> [[String]] -> [[String]]
replaceAll oldThings newThings csvData = map (replaceRow oldThings newThings) csvData

-- Passenger型をリストに変換
-- toList :: Vector a -> [a]
convertToFloatLists :: V.Vector Passenger -> [[Float]]
convertToFloatLists vectorData =
  let passengerList = V.toList vectorData
  in map convertPassengerToList $ filter validPassenger passengerList
  

-- 訓練データのフォーマットを整える
treatData :: FilePath -> IO [[Float]]
treatData filePath = do
  csvData <- BL.readFile filePath -- ファイル読み込み
  -- decodeByName :: FromNamedRecord a => ByteString -> Either String (Header, Vector a)
  case decodeByName csvData of
        Left error -> do
          putStrLn $ "Error parsing CSV" ++ error
          return []
        Right (_, v) -> do
          let columnsToDelete = [0, 3, 8, 10] -- passengerID, name, ticket, cabinの削除
          let floatLists = convertToFloatLists v
          return $ deleteAllColumns floatLists columnsToDelete


       --   let deletedAllColumns = deleteAllColumns (convertToFloatLists v) columnsToDelete -- いらない行の削除
         -- return deletedAllColumns

  
  -- let parsedCsvData = parseCSV csvData -- csvデータをリストに変換



main :: IO ()
main = do
  treatedData <- treatData "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
  print $ take 5 treatedData

      -- 【sex】'male'を0に, 'female'を1に置き換える
    -- let replacedSexData = replaceAll ["male", "female"] ["0", "1"] treatedData
    -- print $ take 5 replacedSexData


    -- 【sex】'male'を0に, 'female'を1に置き換える
    
    -- 【embarked】'Q'を0に, 'S'を1に, 'C'を2に置き換える