import csv
import cv2
import os
import time
import itertools
from pathlib import Path
from models.image_alignment import compare_images
import functions
from match_pairs import superglue_similarity


def save_results_to_csv(final_results, folder1, folder2, output_csv):
    """將匹配結果保存到 CSV 文件"""
    # 确保输出目录存在
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        # 如果是新的文件，寫入標題
        if file.tell() == 0:
            writer.writerow(
                [
                    "Image 1",
                    "Image 2",
                    "Folder 1",
                    "Folder 2",
                    "Rotation",
                    "Keypoints1",
                    "Keypoints2",
                    "Matches",
                ]
            )

        for (image1, image2, rotation, kpts1, kpts2), matches in final_results.items():
            writer.writerow(
                [image1, image2, folder1, folder2, rotation, kpts1, kpts2, matches]
            )


def main():
    datasets_dir = Path("image_1")

    # **设置要遍历的子文件夹范围**
    start_folder = 1  # 例如要从 image_1/3 开始
    end_folder = 180  # 例如要处理到 image_1/180

    # 确保临时目录和结果目录存在
    temp_dir = Path("temp_aligned_images")
    temp_dir.mkdir(exist_ok=True)

    results_dir = Path("test/results/dump_match_pairs_different")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 确保CSV输出目录存在
    csv_output_dir = Path("test/results/pairs_data/matches/different")
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    # **按數字順序排序子文件夾**
    all_folders = sorted(
        [folder for folder in datasets_dir.glob("*") if folder.is_dir()],
        key=lambda f: int(f.name) if f.name.isdigit() else f.name,
    )

    # **筛选出指定范围内的文件夹**
    selected_folders = [
        folder
        for folder in all_folders
        if folder.name.isdigit() and start_folder <= int(folder.name) <= end_folder
    ]

    # 預處理所有圖片（重新命名、轉換格式）
    for folder in selected_folders:  # 只处理指定范围内的文件夹
        functions.rename_files_in_directory(folder)
        functions.convert_folder_jpg_to_png(folder)

    # **篩選有效文件夾（至少含有2張圖片的文件夾）**
    valid_folders = []
    for folder in selected_folders:
        images = list(folder.glob("*.png"))
        if len(images) > 1:  # 只保留至少含有2張圖片的文件夾
            valid_folders.append(folder)
        else:
            print(f"⚠️ 文件夾 {folder.name} 圖片數量不足，跳過")

    # **創建不同文件夾之間的配對（不重複）**
    folder_pairs = list(itertools.combinations(valid_folders, 2))

    # **計算總比對數量**
    total_comparisons = 0
    for folder1, folder2 in folder_pairs:
        images1 = list(folder1.glob("*.png"))
        images2 = list(folder2.glob("*.png"))
        pair_comparisons = len(images1) * len(images2)
        total_comparisons += pair_comparisons

    print(f"\n📂 有效文件夾數量: {len(valid_folders)}")
    print(f"🖼 總圖片比對數量: {total_comparisons}\n")

    processed_count = 0  # 記錄已比對的圖片對數量

    # **遍歷文件夾對**
    for folder_idx, (folder1, folder2) in enumerate(folder_pairs, start=1):

        # 獲取兩個文件夾中的所有PNG圖片
        images1 = sorted([f for f in folder1.glob("*.png")])
        images2 = sorted([f for f in folder2.glob("*.png")])

        # 本輪要處理的圖片對數量
        current_pairs_count = len(images1) * len(images2)

        print(
            f"\n🔄 處理第 {folder_idx}/{len(folder_pairs)} 對文件夾: {folder1.name} 與 {folder2.name}"
        )
        print(f"📊 當前進度: {processed_count}/{total_comparisons}")
        # 構建所有可能的跨文件夾圖片對
        all_pairs = list(itertools.product(images1, images2))
        print(f"    共有 {len(all_pairs)} 對跨文件夾圖片需要比較")

        # 確保結果目錄存在
        folder_pair_dir = f"{folder1.name}_vs_{folder2.name}"
        folder_results_dir = results_dir / folder_pair_dir
        folder_results_dir.mkdir(parents=True, exist_ok=True)

        # 準備記錄配對信息的文件
        pairs_txt_path = temp_dir / "pairs.txt"
        with open(pairs_txt_path, "w") as f:
            for pair_idx, (img1_path, img2_path) in enumerate(all_pairs, start=1):
                # 檢查是否需要跳過某些圖片對（可選）
                print(f"    目前正在比對第 {pair_idx}/{len(all_pairs)} 張圖片對")

                img1, img2 = cv2.imread(str(img1_path)), cv2.imread(str(img2_path))
                if img1 is None or img2 is None:
                    print(f"❌ 無法讀取圖片: {img1_path} 或 {img2_path}")
                    continue

                start_time = time.time()  # 记录处理时间

                # **圖片對齊**
                try:
                    aligned_img1, aligned_img2 = compare_images(img1, img2)
                except Exception as e:
                    print(f"❌ 圖片對齊失敗: {img1_path} 和 {img2_path} - 錯誤: {e}")
                    continue

                # **儲存對齊後的圖片**
                aligned_img1_path = temp_dir / f"aligned_{img1_path.name}"
                aligned_img2_path = temp_dir / f"aligned_{img2_path.name}"
                cv2.imwrite(str(aligned_img1_path), aligned_img1)
                cv2.imwrite(str(aligned_img2_path), aligned_img2)

                if not aligned_img1_path.exists() or not aligned_img2_path.exists():
                    print(
                        f"❌ 無法儲存對齊圖片 {aligned_img1_path} 或 {aligned_img2_path}"
                    )
                    continue

                # 寫入配對信息（在這裡rotation設置為0，因為我們已經對齊了）
                f.write(f"{aligned_img1_path.name} {aligned_img2_path.name} 0 0\n")

                elapsed_time = time.time() - start_time
                if pair_idx % 50 == 0:  # 每50對才顯示處理時間，避免輸出過多
                    print(
                        f"✅ [{folder1.name} vs {folder2.name}] 第 {pair_idx}/{len(all_pairs)} 對，用時 {elapsed_time:.2f} 秒"
                    )

        # **確保 pairs.txt 有內容**
        if pairs_txt_path.stat().st_size == 0:
            print(
                f"⚠️ {folder1.name} 與 {folder2.name} 的 pairs.txt 內容為空，跳過 SuperGlue"
            )
            continue

        # **執行 SuperGlue 匹配**
        print(f"\n🚀 開始執行 SuperGlue 匹配（{folder1.name} vs {folder2.name}）...")
        superglue_start_time = time.time()

        try:
            results = superglue_similarity(
                str(temp_dir),
                str(temp_dir),
                str(pairs_txt_path),
                str(folder_results_dir),
            )

            superglue_elapsed_time = time.time() - superglue_start_time
            print(
                f"🎯 SuperGlue 完成（{folder1.name} vs {folder2.name}），匹配結果數量: {len(results)}，耗時 {superglue_elapsed_time:.2f} 秒\n"
            )

            # **儲存結果**
            save_results_to_csv(
                results,
                folder1.name,
                folder2.name,
                f"{csv_output_dir}/superglue_different_aligned_data.csv",
            )
        except Exception as e:
            print(
                f"❌ SuperGlue 匹配失敗（{folder1.name} vs {folder2.name}）- 錯誤: {e}"
            )
            continue

        # 清理臨時文件，避免佔用過多磁盤空間
        for temp_file in temp_dir.glob("aligned_*.png"):
            try:
                os.remove(temp_file)
            except Exception as e:
                print(f"⚠️ 無法刪除臨時文件 {temp_file} - {e}")

        processed_count += current_pairs_count

    print(f"\n✨ 所有文件夾配對處理完成！總共處理了 {len(folder_pairs)} 對文件夾")


if __name__ == "__main__":
    main()
