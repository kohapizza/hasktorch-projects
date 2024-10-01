import System.IO

main :: IO ()
main = do
    -- ファイルの読み込み
    content <- readFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/review-texts.txt"

    -- 各行をリストに変換
    -- lines :: String -> [String] 
    let allLines = lines content

    -- 最初の50行だけ抽出
    -- take :: Int -> [a] -> [a]
    let first50lines = take 50 allLines

    -- ファイルを書き込み
    -- writeFile :: FilePath -> String -> IO()
    treatedFile <- writeFile "/home/acf16406dh/hasktorch-projects/app/wordEmbeddings/datas/treated-texts.txt" $ unlines first50lines

    print treatedFile