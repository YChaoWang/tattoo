import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
)
import joblib
import glob
import os
from sklearn.utils import compute_class_weight

directory = "有對齊刺青"
version = "v1.0.0"


def read_csv_file(file_path):
    return pd.read_csv(file_path)


def normalize_features(df, scaler=None):
    # 需要標準化的特徵欄位
    features = ["block1_similarity", "block2_similarity", "block3_similarity"]
    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])
    return df, scaler


def prepare_data(df):
    # 提取特徵與標籤
    features = ["block1_similarity", "block2_similarity", "block3_similarity"]
    X = df[features]
    y = df["label"]
    return X, y


def train_svm_with_all_data(X, y):
    pd.set_option("display.max_rows", None)

    print(y.value_counts())  # 顯示各類別樣本數

    # 計算類別權重以解決不平衡問題
    class_weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    class_weight_dict = dict(zip(np.unique(y), class_weights))

    print("\n平衡後的類別權重:")
    for class_label, weight in class_weight_dict.items():
        print(f"類別 {class_label}: {weight:.4f}")

    # 訓練 LinearSVC 模型
    svm = LinearSVC(
        C=1,
        random_state=0,
        class_weight="balanced",
    )
    svm.fit(X, y)

    y_pred = svm.predict(X)

    # 計算評估指標
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average="binary")
    recall = recall_score(y, y_pred, average="binary")
    f1 = f1_score(y, y_pred, average="binary")

    # 計算特徵重要性
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": np.abs(svm.coef_[0])}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    feature_importance["importance"] = (
        feature_importance["importance"] / feature_importance["importance"].sum()
    )
    feature_importance["importance"] = feature_importance["importance"].round(1)

    feature_importance.to_csv(
        f"{version}_feature_similarity_importance_rounded.csv", index=False
    )
    print(f"特徵重要性已儲存至 '{version}_feature_similarity_importance_rounded.csv'")

    # 生成 LinearSVC 分類公式
    w = svm.coef_[0]
    b = svm.intercept_[0]  # 偏置項
    formula = " + ".join(
        [f"{w[i]:.4f} * {X.columns[i]}" for i in range(len(X.columns))]
    )
    print(f"\n{directory} 分類公式: {formula} + {b:.4f} = 0")

    return svm, accuracy, precision, recall, f1, feature_importance, y_pred


if __name__ == "__main__":
    all_data = []

    # 處理負樣本 (不同的圖片)
    diff_distance_files = glob.glob(
        f"{directory}/相似度/不同/feature_diff_similarities_*.csv"
    )
    for file in diff_distance_files:
        try:
            df = read_csv_file(file)
            df["label"] = 0  # 標記為負樣本
            all_data.append(df)
            print(f"處理不同類別的檔案: {file}")
        except FileNotFoundError:
            print(f"找不到檔案，跳過: {file}")

    # 處理正樣本 (相同的圖片)
    for index in range(1, 180):  # 遍歷 1-83 的檔案
        distances_file = f"{directory}/相似度/相同/feature_similarities_{index}.csv"
        try:
            df = read_csv_file(distances_file)
            df["label"] = 1  # 標記為正樣本
            all_data.append(df)
            print(f"處理相同類別的檔案: {distances_file}")
        except FileNotFoundError:
            print(f"找不到檔案，跳過: {distances_file}")

    # 合併所有數據
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df, scaler = normalize_features(combined_df)

    # 準備訓練資料
    X, y = prepare_data(combined_df)

    # 訓練 SVM 模型
    svm_model, accuracy, precision, recall, f1, feature_importance, y_pred = (
        train_svm_with_all_data(X, y)
    )

    # 顯示評估結果
    print("\nSVM 分類器評估指標:")
    print(f"準確率: {accuracy:.4f}")
    print(f"精確率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1 值: {f1:.4f}")

    print("\n混淆矩陣:")
    print(confusion_matrix(y, y_pred))

    print("\n特徵重要性:")
    print(feature_importance)

    # 儲存模型和數據
    joblib.dump(svm_model, f"{version}_svm_pipeline_similarity_only.joblib")
    joblib.dump(scaler, f"{version}_scaler_similarity_only.joblib")
    feature_importance.to_csv(
        f"{version}_feature_importance_similarity_only.csv", index=False
    )
    combined_df.to_csv(f"{version}output_labeled_data_similarity_only.csv", index=False)

    print("模型、特徵重要性和標記數據已儲存。")
