import sys
import os

# Add the parent directory of vgg_compare to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .vgg_same_folder_aligned import process_same_folder_images
from .vgg_different_folder_aligned import process_different_folder_images


def get_image_paths(classifier_type, file_type):
    base_path = "images關防" if classifier_type == "關防" else "images公司"

    if file_type == "相同":
        if classifier_type == "關防":
            indices = range(1, 13)
        else:  # 公司
            indices = range(13, 84)
        image_paths = []
        for i in indices:
            folder_path = os.path.join(base_path, str(i))
            print(f"Checking folder: {folder_path}")  # Debugging statement
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                print(f"Files in folder: {files}")  # Debugging statement
                image_paths.extend(
                    [
                        os.path.join(folder_path, f)
                        for f in files
                        if f.endswith(".jpg") or f.endswith(".png")
                    ]
                )
        print(f"Total images found: {len(image_paths)}")  # Debugging statement
        return image_paths
    else:  # "不同"
        image_paths = []
        for i in range(1, 13) if classifier_type == "關防" else range(13, 84):
            folder_path = os.path.join(base_path, str(i))
            print(f"Checking folder: {folder_path}")  # Debugging statement
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                print(f"Files in folder: {files}")  # Debugging statement
                image_paths.extend(
                    [
                        os.path.join(folder_path, f)
                        for f in files
                        if f.endswith(".jpg") or f.endswith(".png")
                    ]
                )
        print(f"Total images found: {len(image_paths)}")  # Debugging statement
        return image_paths


def main():
    classifier_type = input("請選擇分類類型（關防/公司）：")
    file_type = input("請選擇要分類的文件類型（相同/不同）：")

    if classifier_type not in ["關防", "公司"] or file_type not in ["相同", "不同"]:
        print("輸入無效。請確保選擇正確的分類類型和文件類型。")
        return

    image_paths = get_image_paths(classifier_type, file_type)

    if len(image_paths) < 2:
        print("找不到足夠的圖片進行比較。")
        return

    # 設定處理的文件夾範圍
    start_folder = 1 if classifier_type == "關防" else 13
    end_folder = 12 if classifier_type == "關防" else 83

    if file_type == "相同":
        print("處理相同文件夾內的圖片...")
        process_same_folder_images(
            start_folder,
            end_folder,
            "images關防" if classifier_type == "關防" else "images公司",
        )
    else:
        print("處理不同文件夾間的圖片...")
        process_different_folder_images(
            start_folder,
            end_folder,
            "images關防" if classifier_type == "關防" else "images公司",
        )

    print("所有圖片處理完成。")


if __name__ == "__main__":
    main()
