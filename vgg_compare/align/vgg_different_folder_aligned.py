import numpy as np
import pandas as pd
from PIL import Image
from tf_keras.applications import VGG16
from tf_keras.preprocessing import image
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.models import Model
import os
import sys
import cv2
import time

# 获取 vgg_compare 目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加 vgg_compare 到 sys.path
sys.path.append(parent_dir)
from align.image_alignment import compare_images


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


def process_different_folder_images(start_folder, end_folder, root_dir):
    total_start_time = time.time()

    # 獲取有效資料夾
    valid_folders = get_valid_folders(root_dir, start_folder, end_folder)

    if not valid_folders:
        print("沒有符合條件的資料夾，程序結束。")
        return

    # Specify the layer names
    layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]

    # Load VGG16 model
    base_model = VGG16(weights="imagenet", include_top=False)
    model = Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(name).output for name in layer_names],
    )

    # [原有的輔助函數保持不變]
    def load_and_preprocess_image(img):
        img = Image.fromarray(img).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array

    def extract_features(img):
        img_array = load_and_preprocess_image(img)
        features = model.predict(img_array)
        return [f.flatten() for f in features]

    def calculate_similarity(features1, features2):
        similarities = [
            round(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)), 3)
            for f1, f2 in zip(features1, features2)
        ]
        return similarities

    def save_similarities_to_csv(img1_path, img2_path, similarities, csv_path):
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

    comparison_count = 0
    compared_pairs = set()

    print(f"有效資料夾數量: {len(valid_folders)}")
    print("總比較數: 0", end="", flush=True)

    for folder_idx, folder_path in enumerate(valid_folders, start=1):
        folder_start_time = time.time()
        print(f"\n處理資料夾 {folder_idx}/{len(valid_folders)}: {folder_path}")

        img_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".png"))
        ]

        for other_folder_path in valid_folders:
            if other_folder_path <= folder_path:
                continue

            other_img_paths = [
                os.path.join(other_folder_path, f)
                for f in os.listdir(other_folder_path)
                if f.lower().endswith((".jpg", ".png"))
            ]

            for img1_path in img_paths:
                for img2_path in other_img_paths:
                    pair = tuple(sorted([img1_path, img2_path]))

                    if pair in compared_pairs:
                        continue

                    compared_pairs.add(pair)

                    try:
                        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
                        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

                        if img1 is None or img2 is None:
                            print(f"讀取圖片時發生錯誤: {img1_path}, {img2_path}")
                            continue

                        aligned_img1, aligned_img2 = compare_images(img1, img2)
                        features1 = extract_features(aligned_img1)
                        features2 = extract_features(aligned_img2)
                        similarities = calculate_similarity(features1, features2)

                        csv_path = f"feature_diff_similarities_{os.path.basename(img1_path)[:-4]}.csv"
                        save_similarities_to_csv(
                            img1_path, img2_path, similarities, csv_path
                        )
                    except FileNotFoundError as e:
                        print(e)
                        continue

                    comparison_count += 1
                    sys.stdout.write(f"\r總比較數: {comparison_count}")
                    sys.stdout.flush()

        folder_end_time = time.time()
        print(
            f"\n完成處理資料夾 {folder_idx}/{len(valid_folders)}，耗時 {folder_end_time - folder_start_time:.2f} 秒。"
        )

    total_end_time = time.time()
    print(f"\n最終總比較數: {comparison_count}")
    print(f"總執行時間: {total_end_time - total_start_time:.2f} 秒。")


# 執行函數
process_different_folder_images(
    start_folder=1,
    end_folder=180,
    root_dir="image_1",
)
