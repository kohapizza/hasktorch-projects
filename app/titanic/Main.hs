-- train.csvを扱いやすいデータにする


main :: IO ()
main = do
    csvdata <- readFile "/home/acf16406dh/hasktorch-projects/app/titanic/data/train.csv"
    print csvdata