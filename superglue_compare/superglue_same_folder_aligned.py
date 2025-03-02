import csv
import cv2
import os
import itertools
import time  # 用于计算处理时间
from pathlib import Path
from models.image_alignment import compare_images
import functions
import match_pairs as superglue


def save_results_to_csv(final_results, folder_name, output_csv):
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
                    "Folder",
                    "Rotation",
                    "Keypoints1",
                    "Keypoints2",
                    "Matches",
                ]
            )

        for (image1, image2, rotation, kpts1, kpts2), matches in final_results.items():
            writer.writerow(
                [image1, image2, folder_name, rotation, kpts1, kpts2, matches]
            )


def generate_unique_pairs(image_files):
    """生成不重複的圖片對，確保 a vs b 和 b vs a 只算一次"""
    return list(itertools.combinations(image_files, 2))


def main():
    datasets_dir = Path("/Users/wangyichao/vgg/image_1")

    # **设置要遍历的子文件夹范围**
    start_folder = 3  # 例如要从 image_1/1 开始
    end_folder = 180  # 例如要处理到 image_1/20

    # 确保临时目录和结果目录存在
    temp_dir = Path("temp_aligned_images")
    temp_dir.mkdir(exist_ok=True)

    results_dir = Path("test/results/dump_match_pairs")
    results_dir.mkdir(parents=True, exist_ok=True)

    # 确保CSV输出目录存在
    csv_output_dir = Path("test/results/pairs_data/matches/same")
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

    # **遍歷範圍內的文件夾**
    for folder in selected_folders:
        image_files = sorted([f.name for f in folder.glob("*.png")])  # 确保文件名按顺序
        if not image_files:
            print(f"⚠️ 文件夾 {folder.name} 沒有 PNG 圖片，跳過處理")
            continue

        unique_pairs = generate_unique_pairs(image_files)
        print(f"\n📂 正在處理 {folder.name}，共有 {len(unique_pairs)} 對不重複圖片\n")

        temp_dir = Path("temp_aligned_images")
        temp_dir.mkdir(exist_ok=True)

        pairs_txt_path = temp_dir / "pairs.txt"
        with open(pairs_txt_path, "w") as f:
            for idx, (image1_name, image2_name) in enumerate(unique_pairs, start=1):
                img1_path, img2_path = folder / image1_name, folder / image2_name

                img1, img2 = cv2.imread(str(img1_path)), cv2.imread(str(img2_path))
                if img1 is None or img2 is None:
                    print(f"❌ 無法讀取圖片: {img1_path} 或 {img2_path}")
                    continue

                start_time = time.time()  # 记录处理时间

                # **圖片對齊**
                aligned_img1, aligned_img2 = compare_images(img1, img2)

                # **儲存對齊後的圖片**
                aligned_img1_path = temp_dir / f"aligned_{image1_name}"
                aligned_img2_path = temp_dir / f"aligned_{image2_name}"
                cv2.imwrite(str(aligned_img1_path), aligned_img1)
                cv2.imwrite(str(aligned_img2_path), aligned_img2)

                if not aligned_img1_path.exists() or not aligned_img2_path.exists():
                    print(
                        f"❌ 無法儲存對齊圖片 {aligned_img1_path} 或 {aligned_img2_path}"
                    )
                    continue

                f.write(f"{aligned_img1_path.name} {aligned_img2_path.name} 0 0\n")

                elapsed_time = time.time() - start_time  # 计算时间
                print(
                    f"✅ [Folder {folder.name}] 第 {idx}/{len(unique_pairs)} 對圖片處理完成，用時 {elapsed_time:.2f} 秒"
                )

        # **確保 pairs.txt 有內容**
        if pairs_txt_path.stat().st_size == 0:
            print(f"⚠️ {folder.name} 的 pairs.txt 內容為空，跳過 SuperGlue")
            continue

        # 确保结果目录存在
        folder_results_dir = results_dir / folder.name
        folder_results_dir.mkdir(parents=True, exist_ok=True)

        # **執行 SuperGlue 匹配**
        print(f"\n🚀 開始執行 SuperGlue 匹配（{folder.name}）...")
        superglue_start_time = time.time()

        results = superglue.superglue_similarity(
            str(temp_dir),
            str(temp_dir),
            str(pairs_txt_path),
            f"test/results/dump_match_pairs/{folder.name}",
        )
        print(f"🔍 SuperGlue 结果: {results}")

        superglue_elapsed_time = time.time() - superglue_start_time
        print(
            f"🎯 SuperGlue 完成（{folder.name}），耗時 {superglue_elapsed_time:.2f} 秒\n"
        )

        # **儲存結果**
        save_results_to_csv(
            results,
            folder.name,
            "test/results/pairs_data/matches/same/all_data.csv",
        )


if __name__ == "__main__":
    main()
