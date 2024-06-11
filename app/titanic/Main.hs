import Data.List.Split (splitOn)
import System.IO
import Data.List.Utils (replace)

-- passengerld： 乗客者ID
-- survived：生存状況（0＝死亡、1＝生存）
-- pclass： 旅客クラス（1＝1等、2＝2等、3＝3等）。裕福さの目安となる
-- name： 乗客の名前
-- sex： 性別（male＝男性、female＝女性）
-- age： 年齢。一部の乳児は小数値
-- sibsp： タイタニック号に同乗している兄弟（Siblings）や配偶者（Spouses）の数
-- parch： タイタニック号に同乗している親（Parents）や子供（Children）の数
-- ticket： チケット番号
-- fare： 旅客運賃
-- cabin： 客室番号
-- embarked： 出港地（C＝Cherbourg：シェルブール、Q＝Queenstown：クイーンズタウン、S＝Southampton：サウサンプトン）

-- csvデータをリストに変換
parseCSV :: String -> [[String]]
-- lines csvData :: [String]
parseCSV csvData = map (splitOn ",") (lines csvData)

-- リストからいらない列(index番目)を消す
deleteColumn :: Int -> [String] -> [String]
deleteColumn index row = take index row ++ drop (index + 1) row

-- リストから複数のいらない列を消す
deleteColumns :: [Int] -> [String] -> [String]
deleteColumns idxs row = foldl (\r i -> deleteColumn i r) row (reverse (sorted idxs))
  where
    sorted = map fst . filter snd . zip [0..] . flip map idxs . flip elem

-- 全ての行から複数の列を削除
deleteAllColumns :: [[String]] -> [Int] -> [[String]]
deleteAllColumns parsedCsvData columnsToDelete = map (deleteColumns columnsToDelete) parsedCsvData

-- 全ての行の'oldThing'を'newNum'に置き換える
-- replace :: Eq a => [a] -> [a] -> [a] -> [a]
replaceAll :: [String] -> [String] -> [[String]] -> [[String]]
replaceAll oldThing newNum csvData = map (replace oldThing newNum) csvData

main :: IO ()
main = do
    let columnsToDelete = [3, 8, 10]
    -- readFile :: FilePath -> IO String
    -- csvData :: String
    csvData <- readFile "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
    print $ take 5 $ deleteAllColumns (parseCSV csvData) columnsToDelete -- name, cabin, and ticketsの削除


    -- 【sex】'male'を0に, 'female'を1に置き換える
    
    -- 【embarked】'Q'を0に, 'S'を1に, 'C'を2に置き換える
    


