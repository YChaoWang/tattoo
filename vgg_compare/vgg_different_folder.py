import numpy as np
import pandas as pd
from PIL import Image
from tf_keras.applications import VGG16
from tf_keras.preprocessing import image
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.models import Model
import os
import sys
import time

# 加載 VGG16 模型並定義用於特徵提取的層
base_model = VGG16(weights="imagenet", include_top=False)
layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]
model = Model(
    inputs=base_model.input,
    outputs=[base_model.get_layer(name).output for name in layer_names],
)


def load_and_preprocess_image(img_path):
    """加載並預處理圖像"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def extract_features(img_path):
    """提取圖像特徵"""
    img_array = load_and_preprocess_image(img_path)
    features = model.predict(img_array)
    return [f.flatten() for f in features]


def calculate_similarity(features1, features2):
    """計算特徵之間的相似度"""
    similarities = [
        round(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)), 3)
        for f1, f2 in zip(features1, features2)
    ]
    return similarities


def save_feature_similarities_to_csv(img1_path, img2_path, similarities, csv_path):
    """將特徵相似度保存到 CSV 檔案"""
    df = pd.DataFrame(
        {
            "img1": [img1_path],
            "img2": [img2_path],
            "block1_similarity": [similarities[0]],
            "block2_similarity": [similarities[1]],
            "block3_similarity": [similarities[2]],
        }
    )

    if os.path.isfile(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"已保存 {img1_path} 和 {img2_path} 的相似度到 {csv_path}")


def get_valid_folders(root_dir, start_folder, end_folder):
    """獲取至少包含 2 張圖片的資料夾"""
    valid_folders = []

    for folder_num in range(start_folder, end_folder + 1):
        folder_path = os.path.join(root_dir, str(folder_num))

        if not os.path.isdir(folder_path):
            continue  # 跳過不存在的資料夾

        img_files = [
            f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png"))
        ]

        if len(img_files) > 1:  # 只保留至少 2 張圖的資料夾
            valid_folders.append(folder_path)

    return valid_folders


def compare_images_in_folders(root_dir="images", start_folder=1, end_folder=32):
    """比較資料夾中的所有圖像，略過只有 1 張圖片的資料夾"""

    valid_folders = get_valid_folders(root_dir, start_folder, end_folder)

    if not valid_folders:
        print("沒有符合條件的資料夾，程序結束。")
        return

    comparison_count = 0
    compared_pairs = set()  # 用於追蹤已比較的圖像對
    total_start_time = time.time()

    print(f"有效資料夾數量: {len(valid_folders)}")
    print("總比較數: 0", end="", flush=True)

    for folder_idx, folder_path in enumerate(valid_folders, start=1):
        start_time = time.time()
        print(f"\n處理資料夾 {folder_idx}/{len(valid_folders)}: {folder_path}")

        img_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png"))
        ]

        for other_folder_path in valid_folders:
            if other_folder_path <= folder_path:
                continue  # 跳過已處理過的資料夾或相同的資料夾

            other_img_paths = [
                os.path.join(other_folder_path, f)
                for f in os.listdir(other_folder_path)
                if f.lower().endswith((".jpg", ".png"))
            ]

            for img1_path in img_paths:
                for img2_path in other_img_paths:
                    # 創建排序過的圖像對來確保唯一性
                    pair = tuple(sorted([img1_path, img2_path]))

                    if pair in compared_pairs:
                        continue  # 跳過已比較的圖像對

                    compared_pairs.add(pair)

                    features1 = extract_features(img1_path)
                    features2 = extract_features(img2_path)
                    similarities = calculate_similarity(features1, features2)

                    csv_path = f"feature_diff_similarities_{os.path.basename(img1_path)[:-4]}.csv"
                    save_feature_similarities_to_csv(
                        img1_path, img2_path, similarities, csv_path
                    )

                    comparison_count += 1
                    sys.stdout.write(f"\r總比較數: {comparison_count}")
                    sys.stdout.flush()

        end_time = time.time()
        print(
            f"\n完成處理資料夾 {folder_idx}/{len(valid_folders)}，耗時 {end_time - start_time:.2f} 秒。"
        )

    total_end_time = time.time()
    print(f"\n最終總比較數: {comparison_count}")
    print(f"總執行時間: {total_end_time - total_start_time:.2f} 秒。")


# 執行函數
compare_images_in_folders(
    root_dir="image_1",
    start_folder=1,
    end_folder=180,
)
