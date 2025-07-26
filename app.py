import pandas as pd
from collections import Counter
import streamlit as st
import joblib
import numpy as np
import random

st.set_page_config(page_title="ミニロト予測Bot", page_icon="🎯")

st.title("🎯 ミニロト予測Bot")

# --- データ読み込み ---
df = pd.read_csv("miniloto_all.csv")

# --- 特徴量生成（AI予測の準備） ---
draws = df[["数字1", "数字2", "数字3", "数字4", "数字5"]].values.tolist()
recent10 = draws[-10:]
flat = [num for draw in recent10 for num in draw]
counter = {n: flat.count(n) for n in range(1, 32)}

X_pred = pd.DataFrame({
    "number": range(1, 32),
    "count_last10": [counter[n] for n in range(1, 32)]
})

# --- モデル読み込み・予測 ---
model = joblib.load("miniloto_model.pkl")
proba = model.predict_proba(X_pred)[:, 1]

result_df = pd.DataFrame({
    "数字": range(1, 32),
    "出現確率": np.round(proba, 4)
}).sort_values("出現確率", ascending=False).reset_index(drop=True)

# --- 🎰 最初にAI予測による5口分を表示 ---
st.subheader("🎰 AI予測に基づくおすすめ数字（5口）")

if st.button("🔁 もう一度生成"):
    st.rerun()

候補数字 = result_df["数字"].tolist()
上位候補 = 候補数字[:20]

suggestions = []
for _ in range(5):
    nums = sorted(random.sample(上位候補, 5))
    suggestions.append(nums)

for i, s in enumerate(suggestions, 1):
    formatted = "・".join([str(n) for n in s])
    st.write(f"{i}番口： 🎯 {formatted}")

# --- 🔢 最新の当選データ表示 ---
st.subheader("📅 最新の当選データ")
st.dataframe(df.tail(5))

# --- 頻出数字ベースの予測表示（見栄え改善） ---
numbers = df[["数字1", "数字2", "数字3", "数字4", "数字5"]].values.flatten()
counter = Counter(numbers)
top5 = [int(num) for num, _ in counter.most_common(5)]

st.subheader("✨ 頻繁に出てくる数字ベースのおすすめ5桁")
formatted_freq = "・".join([str(n) for n in sorted(top5)])
st.write(f"🎯 {formatted_freq}")

# --- 出現確率ランキング表示 ---
st.subheader("🧠 AIモデルによる出現確率ランキング")
st.dataframe(result_df.head(10))
