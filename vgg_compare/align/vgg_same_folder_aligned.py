import sys
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from PIL import Image
from tf_keras.applications import VGG16
from tf_keras.preprocessing import image
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.models import Model
import itertools
import time
from tqdm import tqdm
import cv2

# 获取 vgg_compare 目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 添加 vgg_compare 到 sys.path
sys.path.append(parent_dir)

from align.image_alignment import compare_images


def process_same_folder_images(start_folder, end_folder, root_dir):
    total_start_time = time.time()
    comparison_count = 0

    # Specify the layer names
    layer_names = [
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
    ]

    # Load VGG16 model (excluding the top fully connected layers)
    base_model = VGG16(weights="imagenet", include_top=False)

    # Get the output of the specified layers
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create a new model to extract features from the specified layers
    model = Model(inputs=base_model.input, outputs=layer_outputs)

    def extract_features(img, img_path, model, layer_names):
        img = Image.fromarray(img).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Extract features from each layer
        features = model.predict(img_array)

        # Return a dictionary with features for each layer
        feature_dict = {
            name: features[i].flatten() for i, name in enumerate(layer_names)
        }

        return feature_dict

    def calculate_similarity(features1, features2):
        similarities = {}
        for layer_name in features1.keys():
            f1, f2 = features1[layer_name], features2[layer_name]
            similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            similarities[layer_name] = round(similarity, 3)
        return similarities

    def save_similarities_to_csv(img1_path, img2_path, similarities, csv_path):
        # Create a DataFrame from the similarities dictionary
        df = pd.DataFrame(
            {
                "img1": [img1_path],
                "img2": [img2_path],
                "block1_similarity": [similarities["block1_conv2"]],
                "block2_similarity": [similarities["block2_conv2"]],
                "block3_similarity": [similarities["block3_conv3"]],
            }
        )

        # Check if file exists
        if os.path.isfile(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, mode="w", header=True, index=False)

    def convert_png_to_jpg(image_path):
        if image_path.endswith(".png"):
            img = Image.open(image_path).convert("RGB")
            new_image_path = image_path.replace(".png", ".jpg")
            img.save(new_image_path, "JPEG")
            os.remove(image_path)
            return new_image_path
        return image_path

    # Count valid folders
    valid_folders = [
        folder_num
        for folder_num in range(start_folder, end_folder + 1)
        if os.path.isdir(os.path.join(root_dir, str(folder_num)))
    ]
    print(f"有效資料夾數量: {len(valid_folders)}")
    print("總比較數: 0", end="", flush=True)

    # Process each folder
    for folder_idx, folder_num in enumerate(valid_folders, start=1):
        folder_start_time = time.time()
        folder_path = os.path.join(root_dir, str(folder_num))
        print(f"\n處理資料夾 {folder_idx}/{len(valid_folders)}: {folder_path}")

        # Get all jpg and png images in the folder
        img_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        if not img_paths:
            print(f"資料夾 {folder_num} 中沒有找到圖片")
            continue

        # Convert png images to jpg
        img_paths = [convert_png_to_jpg(img_path) for img_path in img_paths]

        # Define CSV file path for this folder
        csv_path = f"feature_similarities_{folder_num}.csv"

        # Compare each pair of images
        for img1_path, img2_path in itertools.combinations(img_paths, 2):
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            if img1 is None or img2 is None:
                print(f"讀取圖片時發生錯誤: {img1_path}, {img2_path}")
                continue

            # Align images
            aligned_img1, aligned_img2 = compare_images(img1, img2)

            # Extract features and calculate similarity
            features1 = extract_features(aligned_img1, img1_path, model, layer_names)
            features2 = extract_features(aligned_img2, img2_path, model, layer_names)
            similarities = calculate_similarity(features1, features2)

            # Save similarities to CSV
            save_similarities_to_csv(img1_path, img2_path, similarities, csv_path)

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


# 示例使用
process_same_folder_images(
    start_folder=1,
    end_folder=180,
    root_dir="image_1",
)
