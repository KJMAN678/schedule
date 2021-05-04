### ライブラリのインポート
from itertools import product
import pandas as pd
import streamlit as st
from mip import BINARY, Model, maximize, xsum, OptimizationStatus
from more_itertools import pairwise, windowed
import base64

### サンプルデータのダウンロード
download=st.sidebar.button('サンプルCSVをダウンロード')
if download:
  'サンプルCSV内訳'
  df_download= pd.DataFrame(
    [
     ["佐藤", "休", "", "", "", "", "", "", ""],
     ["田中", "", "休", "", "", "", "", "", ""],
     ["鈴木", "", "", "休", "", "", "", "", ""],
     ["高橋", "", "", "", "", "", "", "", "休"],
    ]
)
  df_download.columns=["Name", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
  df_download
  csv = df_download.to_csv(index=False)
  b64 = base64.b64encode(csv.encode()).decode()  # some strings
  linko= f'<a href="data:file/csv;base64,{b64}" download="sample.csv">クリックしてcsvをダウンロード</a>'
  st.markdown(linko, unsafe_allow_html=True)

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

### 表示するステータス
status = ""
if m.status == OptimizationStatus.OPTIMAL:
  status = "最適解が算定されました"
elif m.status == OptimizationStatus.INFEASIBLE:
  status = "実行不可能でした"
elif m.status == OptimizationStatus.UNBOUNDED:
  status = "解が無限に存在します"
elif m.status == OptimizationStatus.FEASIBLE:
  status = "整数の実行可能解が見つかりましたが、これが最適解であるかどうかを判断する前に検索が中断されました。"
elif m.status == OptimizationStatus.INT_INFEASIBLE:
  status = "整数問題は実行不可能でした"
elif m.status == OptimizationStatus.NO_SOLUTION_FOUND:
  status = "整数の実行可能解が見つかりませんでした"
elif m.status == OptimizationStatus.LOADED:
  status = "問題はロードされましたが、最適解は実行されませんでした"
elif m.status == OptimizationStatus.CUTOFF:
  status = "現在のカットオフに対して実行可能な解決策はありません"
elif m.status == OptimizationStatus.ERROR:
  status = "エラーが発生しました"

f"""
# 看護師のスケジュール作成
## 実行結果
- ステータス：{status}
- 希望をかなえた数：{m.objective.x}
""" # Markdownとして解釈される

f = lambda s: f"color: {'red' * (s == '休')}"
st.dataframe(res.style.applymap(f), width=500, height=300) # 計算結果を出力する