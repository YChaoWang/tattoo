import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

print("Current Working Directory:", os.getcwd())
# 設置字體和解決負號顯示問題
plt.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei"
]  # 使用微軟正黑體（或其他支持中文的字體）
plt.rcParams["axes.unicode_minus"] = False  # 解決負號顯示問題

# 更新文件路徑，去掉 'company' 這個資料夾
file_path_same = "results/pairs_data/matches/same/same_data_0.6.csv"

data_same = pd.read_csv(file_path_same)

# 定義新的 'Matches' 範圍
specified_matches = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

# 計算大於每個指定 matches 數量的資料數量 (相同資料夾)
cumulative_counts_same = [
    (data_same["Matches"] > match).sum() for match in specified_matches
]
cumulative_percentages_same = np.round(
    (np.array(cumulative_counts_same) / len(data_same)) * 100, 2
)

# 繪製折線圖
plt.figure(figsize=(10, 6))
plt.plot(
    specified_matches,
    cumulative_percentages_same,
    marker="o",
    linestyle="-",
    color="b",
    label="相同資料夾",
)

plt.xlabel("Matches 數量")
plt.ylabel("大於該數量的資料百分比 (%)")
plt.title(
    "原始 大於指定 Matches 數量的資料百分比（相同資料夾） Superglue閥值0.6 前處理"
)
plt.xlim(250, 360)  # 設定 x 軸範圍
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()  # 顯示圖例

# 確保目錄存在
output_dir = "results/pairs_data/matches/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 儲存圖片
plt.savefig(os.path.join(output_dir, "原始相同數量閥值0.6_指定點數量.png"), dpi=300)
plt.show()

# 獲取總資料數
total_data_same = len(data_same)

# 更新資料轉換為 DataFrame
output_data_same = {
    "Matches數量": specified_matches,
    "大於該數量的資料數量": cumulative_counts_same,
    "總資料數": [total_data_same] * len(specified_matches),
    "大於該數量的資料百分比": cumulative_percentages_same,
}

# 創建 DataFrame
df_same = pd.DataFrame(output_data_same)

# 儲存 CSV 檔案
df_same.to_csv(
    os.path.join(output_dir, "對齊相同類別0.6_指定點數量.csv"),
    index=False,
    encoding="utf-8-sig",
)


# 使用 Pandas 的 plot_table 來生成資料表格的圖片
def save_dataframe_as_image(df, path, title):
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.4))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, pad=60)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# 更新圖片儲存路徑，去掉 'company'
image_output_dir = "results/pairs_data/matches/image/0.6 對齊/"
if not os.path.exists(image_output_dir):
    os.makedirs(image_output_dir)

save_dataframe_as_image(
    df_same,
    os.path.join(image_output_dir, "對齊相同類別0.6_指定點數量.png"),
    "對齊相同類別0.6_指定點數量",
)
