import System.IO
import Data.List.Split

main :: IO ()
main = do
    -- textFilePathに関して
    -- ファイルの読み込み
    content <- readFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/review-texts.txt"

    -- 各行をリストに変換
    -- lines :: String -> [String] 
    let allLines = lines content

    -- 最初の5行だけ抽出
    -- take :: Int -> [a] -> [a]
    let first5lines = take 5 allLines

    -- ファイルを書き込み
    -- writeFile :: FilePath -> String -> IO()
    treatedFile <- writeFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/treated-texts.txt" $ unlines first5lines


    -- wordLstPathに関して
    -- ファイルの読み込み
    wordLstContent <- readFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/treated-texts.txt"

    -- 各行をリストに変換
    -- allLstLines :: [String]
    let allLstLines = lines wordLstContent

    -- 各リストの要素に対して、単語で区切る
    -- splitOn :: Eq a => [a] -> [a] -> [[a]]
    let wordLists = map (splitOn " ") allLstLines

    -- リストを平坦化
    -- wordList :: [String]
    let wordList = concat wordLists

    -- ファイルを書き込み
    -- writeFile :: FilePath -> String -> IO()
    wordListFile <- writeFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/wordLists.txt" $ unlines wordList

    print wordListFile