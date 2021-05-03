### ライブラリのインポート
from itertools import product
import pandas as pd
import streamlit as st
from mip import BINARY, Model, maximize, xsum
from more_itertools import pairwise, windowed

### テーブルデータの作成
shifts = ["日", "夜", "休"] # シフトリスト
wish = st.sidebar.file_uploader("希望シフト") # csvファイルのアップローダー
dfws = pd.read_csv(wish or "wish.csv") # 希望データ
days = dfws.columns[1:] # 日付リスト
dffx = dfws.melt("Name", days, "Day", "Shift").dropna()
st.sidebar.dataframe(dffx) # 希望シフトをサイドバーに出力する

### 数理最適化処理
d = product(dfws.Name, days, shifts)
df = pd.DataFrame(d, columns=dffx.columns)
m = Model()
x = m.add_var_tensor((len(df),), "x", var_type=BINARY)
df["Var"] = x
m.objective = maximize(xsum(dffx.merge(df).Var))

for _, gr in df.groupby(["Name", "Day"]):
    m += xsum(gr.Var) == 1 # 看護師と日付の組み合わせごとにシフトは1つ
    
for _, gr in df.groupby("Day"):
    m += xsum(gr[gr.Shift == "日"].Var) >= 2 # 日付ごとに日勤は2以上
    m += xsum(gr[gr.Shift == "夜"].Var) >= 1 # 日付ごとに夜勤は1以上

### クエリ
q1 = "(Day == @d1 & Shift == '夜')|"
q2 = "(Day == @d2 & Shift != '休')"
q3 = "Day in @dd & Shift == '休'"

for _, gr in df.groupby("Name"):
    m += xsum(gr[gr.Shift == "日"].Var) <= 4 # 看護婦ごとに日勤は4以上
    m += xsum(gr[gr.Shift == "夜"].Var) <= 2 # 看護婦ごとに夜勤は2以上
    
    for d1, d2 in pairwise(days):
        m += xsum(gr.query(q1 + q2).Var) <= 1 # 夜勤と翌日休みはどちらかだけ
    
    for dd in windowed(days, 4):
        m += xsum(gr.query(q3).Var) >= 1  # 4連続勤務のうち休みは1日以上
        
m.optimize()
df["Val"] = df.Var.astype(float)
res = df[df.Val > 0]
res = res.pivot_table("Shift", "Name", "Day", "first")

f"""
# 看護師のスケジュール作成
## 実行結果
- ステータス：{m.status}
- 希望をかなえた数：{m.objective.x}
""" # Markdownとして解釈される

f = lambda s: f"color: {'red' * (s == '休')}"
st.dataframe(res.style.applymap(f)) # 計算結果を出力する