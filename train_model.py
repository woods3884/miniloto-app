import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

# CSV読み込み
df = pd.read_csv("miniloto_all.csv")

# 数字1〜5を1列に展開
draws = df[["数字1", "数字2", "数字3", "数字4", "数字5"]]
drawn_numbers = draws.values.tolist()

# 特徴量作成
data = []
for i in range(10, len(drawn_numbers)):  # 最低10回は蓄積したい
    recent = drawn_numbers[i-10:i]  # 直近10回分
    flat = [num for draw in recent for num in draw]
    counter = {n: flat.count(n) for n in range(1, 32)}

    this_draw = drawn_numbers[i]
    for n in range(1, 32):
        data.append({
            "number": n,
            "count_last10": counter[n],             # 直近10回の出現回数
            "target": 1 if n in this_draw else 0     # この回に出たか
        })

df_feat = pd.DataFrame(data)

# 学習
X = df_feat[["number", "count_last10"]]
y = df_feat["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# 保存
import joblib
joblib.dump(model, "miniloto_model.pkl")
print("✅ モデルを保存しました：miniloto_model.pkl")
