import pandas as pd
from collections import Counter

# CSVを読み込み
df = pd.read_csv("miniloto_all.csv")

# 本数字だけを抽出
numbers = df[["数字1", "数字2", "数字3", "数字4", "数字5"]].values.flatten()

# 出現頻度をカウント
counter = Counter(numbers)

# 頻出ランキング上位から5つ選ぶ（予測として）
predicted = [num for num, _ in counter.most_common(5)]

print("おすすめ数字（頻出ベース）:", predicted)
