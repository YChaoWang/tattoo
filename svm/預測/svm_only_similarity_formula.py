import numpy as np
import pandas as pd
import os
from collections import Counter

# 定義分類公式的權重和偏置項，僅保留小數點後四位的版本
classifier = {
    "weights_4": np.array([0.3161, -0.3982, 0.3835]),
    "bias_4": -0.1757,
}


def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        return df[
            [
                "img1",
                "img2",
                "block1_similarity",
                "block2_similarity",
                "block3_similarity",
            ]
        ]
    except Exception as e:
        print(f"讀取文件 {file_path} 時發生錯誤：{e}")
        return None


def classify_with_formula(sample, weights, bias):
    linear_combination = np.dot(sample, weights) + bias
    return 1 if linear_combination >= 0 else 0, linear_combination


def process_dataframe(df, weights, bias, predictions, confidence_scores, image_pairs):
    print(f"處理數據形狀：{df.shape}")
    for _, row in df.iterrows():
        sample = row[
            ["block1_similarity", "block2_similarity", "block3_similarity"]
        ].values
        prediction, confidence = classify_with_formula(sample, weights, bias)
        predictions.append(prediction)
        confidence_scores.append(confidence)
        image_pairs.append((row["img1"], row["img2"]))


def classify_data_with_formula(file_type, weights, bias):
    predictions = []
    confidence_scores = []
    image_pairs = []

    folder = "有對齊刺青"

    if file_type == "相同":
        indices = range(1, 180)  # 用原來的範圍，不再根據分類來區分

        for index in indices:
            file_path = f"{folder}/相似度/{file_type}/feature_similarities_{index}.csv"
            df = read_csv_file(file_path)
            if df is None:
                continue
            process_dataframe(
                df, weights, bias, predictions, confidence_scores, image_pairs
            )
    else:  # "不同"
        diff_filenames = [
            f
            for f in os.listdir(f"{folder}/相似度/不同")
            if f.startswith("feature_diff_similarities_")
        ]
        for filename in diff_filenames:
            file_path = f"{folder}/相似度/不同/{filename}"
            df = read_csv_file(file_path)
            if df is None:
                continue
            process_dataframe(
                df, weights, bias, predictions, confidence_scores, image_pairs
            )

    return predictions, confidence_scores, image_pairs


def run_classification(folder, file_type):
    weights = classifier["weights_4"]
    bias = classifier["bias_4"]

    print(f"\n使用精度 4 進行分類：")
    print(f"權重：{weights}")
    print(f"偏置：{bias}")

    predictions, confidence_scores, image_pairs = classify_data_with_formula(
        file_type, weights, bias
    )

    if predictions:
        prediction_counts = Counter(predictions)
        print(f"分類為 0（不同）的樣本數：{prediction_counts[0]}")
        print(f"分類為 1（相同）的樣本數：{prediction_counts[1]}")

        results_df = pd.DataFrame(
            {
                "img1": [pair[0] for pair in image_pairs],
                "img2": [pair[1] for pair in image_pairs],
                "prediction": predictions,
                "confidence": confidence_scores,
            }
        )
        output_filename = f"classification_{folder}_{file_type}results_similarity_only_precision_4.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"結果已保存至 {output_filename}")
    else:
        print("沒有結果可處理。請檢查您的輸入和文件路徑。")


if __name__ == "__main__":
    file_type = input("請選擇要分類的文件類型（相同/不同）：")
    folder = "有對齊刺青"
    if file_type not in ["相同", "不同"]:
        print("輸入無效。請確保選擇正確的文件類型。")
    else:
        run_classification(folder, file_type)
