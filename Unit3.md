# Set-up of supervised learning; Regression
1. 監督學習
    1. 訓練術據由樣本和標籤組成((sample, label)或(x, y)), 目的是找出發法預測新樣本的標籤
    2. 監督學習用來解決兩類問題
        1. 回歸: y是連續的
        2. 分類: y是離散的, 例如類別
2. 線性回歸
    1. y = w^tx + e
    2. 學習是找出適當的w, 讓e誤差越小越好
3. 廣義線性回歸
    1. 引入基函數, 使函數維持線性, 但在實際的d維成為非線性函數
4. 正則化最小二乘法
    1. 用於緩和過度擬合


# Classification; Density estimation
1. 參數化
    1. 