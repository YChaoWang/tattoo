import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from PIL import Image
from tf_keras.applications import VGG16
from tf_keras.preprocessing import image
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.models import Model
import os
import itertools
import time
from tqdm import tqdm
from pathlib import Path


def process_images(start_folder, end_folder, root_dir):
    start_time = time.time()

    # 指定要提取特徵的 VGG16 卷積層
    layer_names = [
        "block1_conv2",
        "block2_conv2",
        "block3_conv3",
    ]

    # 加載 VGG16 模型（不包含頂部分類層）
    base_model = VGG16(weights="imagenet", include_top=False)
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(inputs=base_model.input, outputs=layer_outputs)

    def extract_features(img_path, model, layer_names):
        img_start_time = time.time()

        # 加載並預處理圖片
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 提取圖片特徵
        features = model.predict(img_array)
        feature_dict = {
            name: features[i].flatten() for i, name in enumerate(layer_names)
        }

        img_end_time = time.time()
        print(
            f"特徵提取時間 {os.path.basename(img_path)}: {img_end_time - img_start_time:.3f} 秒"
        )

        return feature_dict

    def calculate_similarity(features1, features2):
        sim_start_time = time.time()

        # 計算兩張圖片之間的餘弦相似度
        similarities = {}
        for layer_name in features1.keys():
            f1, f2 = features1[layer_name], features2[layer_name]
            similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
            similarities[layer_name] = round(similarity, 3)

        sim_end_time = time.time()
        print(f"相似度計算時間: {sim_end_time - sim_start_time:.3f} 秒")

        return similarities

    def save_similarities_to_csv(img1_path, img2_path, similarities, csv_path):
        # 將計算出的相似度保存到 CSV 檔案
        df = pd.DataFrame(
            {
                "img1": [img1_path],
                "img2": [img2_path],
                "block1_similarity": [similarities["block1_conv2"]],
                "block2_similarity": [similarities["block2_conv2"]],
                "block3_similarity": [similarities["block3_conv3"]],
            }
        )

        # 如果 CSV 檔案已存在則追加，否則新建文件
        if os.path.isfile(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, mode="w", header=True, index=False)

    # **按數字順序排序子文件夾**
    all_folders = sorted(
        [folder for folder in Path(root_dir).glob("*") if folder.is_dir()],
        key=lambda f: int(f.name) if f.name.isdigit() else f.name,
    )

    # **篩選出指定範圍內的文件夾**
    selected_folders = [
        folder
        for folder in all_folders
        if folder.name.isdigit() and start_folder <= int(folder.name) <= end_folder
    ]

    # 初始化 tqdm 進度條
    progress_bar = tqdm(selected_folders, desc="處理文件夾", position=0)

    for folder_path in progress_bar:
        folder_num = int(folder_path.name)
        progress_bar.set_description(f"處理文件夾 {folder_num}")

        # 獲取當前文件夾內的所有圖片
        img_paths = sorted(
            [f for f in folder_path.glob("*.jpg")]
            + [f for f in folder_path.glob("*.png")],
            key=lambda x: x.name,
        )

        # 定義當前文件夾的 CSV 檔案名稱
        csv_path = f"feature_similarities_{folder_num}.csv"

        # 計算每對圖片的特徵相似度
        for img1_path, img2_path in itertools.combinations(img_paths, 2):
            features1 = extract_features(str(img1_path), model, layer_names)
            features2 = extract_features(str(img2_path), model, layer_names)
            similarities = calculate_similarity(features1, features2)
            save_similarities_to_csv(
                str(img1_path), str(img2_path), similarities, csv_path
            )

    end_time = time.time()
    print(f"總執行時間: {end_time - start_time:.3f} 秒")


# 示例使用
process_images(
    start_folder=1,
    end_folder=180,
    root_dir="image_1",
)
