import superglue_compare.match_pairs as superglue
import csv
from pathlib import Path
from superglue_compare import pairs_txt as pairs
from tools import functions
import cv2


def save_results_to_csv(final_results, folder_name, output_csv):
    with open(output_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(
                ["Image 1", "Image 2", "Folder", "Keypoints1", "Keypoints2", "Matches"]
            )
        for (image1, image2, kpts1, kpts2), matches in final_results.items():
            writer.writerow([image1, image2, folder_name, kpts1, kpts2, matches])


def main():
    datasets_dir = Path("datasets")
    all_folders = list(datasets_dir.iterdir())

    # 預處理所有圖片（重新命名、轉換格式）
    for folder in all_folders:
        functions.rename_files_in_directory(folder)
        functions.convert_folder_jpg_to_png(folder)

    # 進行圖片匹配（先對齊，再SuperGlue）
    for idx, folder in enumerate(all_folders[:-1]):
        for image_path in folder.iterdir():
            image_name = image_path.name
            img1 = cv2.imread(str(image_path))  # 讀取圖片

            for second_folder in all_folders[idx + 1 :]:
                for second_image_path in second_folder.iterdir():
                    second_image_name = second_image_path.name
                    img2 = cv2.imread(str(second_image_path))  # 讀取圖片

                    # **圖片對齊**
                    aligned_img1, aligned_img2 = functions.compare_images(img1, img2)

                    # **儲存對齊後的圖片到臨時檔案**
                    temp_dir = Path("temp_aligned_images")
                    temp_dir.mkdir(exist_ok=True)  # 確保目錄存在
                    aligned_img1_path = temp_dir / f"aligned_{image_name}"
                    aligned_img2_path = temp_dir / f"aligned_{second_image_name}"
                    cv2.imwrite(str(aligned_img1_path), aligned_img1)
                    cv2.imwrite(str(aligned_img2_path), aligned_img2)

                    # **生成 SuperGlue 匹配對**
                    pairs_file = pairs(
                        aligned_img1_path.name, aligned_img2_path.name, temp_dir
                    )

                    # **進行 SuperGlue 匹配**
                    results = superglue.superglue_similarity(
                        str(temp_dir),
                        str(temp_dir),
                        pairs_file,
                        f"results/dump_match_pairs/{folder.name}",
                    )

                    # **儲存結果**
                    save_results_to_csv(
                        results,
                        second_folder.name,
                        "test/results/比對資料/matches/different/combined.csv",
                    )


if __name__ == "__main__":
    main()
