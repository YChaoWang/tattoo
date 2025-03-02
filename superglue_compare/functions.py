import csv
from PIL import Image
import os
import re


def convert_folder_jpg_to_png(folder_path):
    # 確認資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    # 確認資料夾內有無 JPG 檔案
    jpg_files = [
        f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".jpeg")
    ]
    if not jpg_files:
        print(f"No JPG files found in folder {folder_path}.")
        return

    # 迭代資料夾內的每個 JPG 檔案
    for jpg_file in jpg_files:
        jpg_path = os.path.join(folder_path, jpg_file)

        # 開啟並轉換 JPG 檔案為 PNG
        img = Image.open(jpg_path)
        png_file = os.path.splitext(jpg_file)[0] + ".png"
        png_path = os.path.join(folder_path, png_file)
        img.save(png_path, "PNG")

        # 刪除原始 JPG 檔案
        os.remove(jpg_path)
        print(f"Converted {jpg_file} to {png_file} and deleted {jpg_file}")


def save_results_to_csv(final_results, output_csv):
    print(f"Saving results to {output_csv}")  # Debug print
    try:
        with open(output_csv, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Image",
                    "Matched Image",
                    "Folder",
                    "Rotation",
                    "Keypoints0",
                    "Keypoints1",
                    "Matched Keypoints",
                    "Percentage Similarity",
                ]
            )

            for image, matches in final_results.items():
                print(f"Processing matches for {image}")  # Debug print
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 3:
                        (
                            (matched_image, folder, rotate, kpts0, kpts1),
                            mkpts0,
                            percentage_similarity,
                        ) = match
                        writer.writerow(
                            [
                                image,
                                matched_image,
                                folder,
                                rotate,
                                kpts0,
                                kpts1,
                                mkpts0,  # This is already the number of matched keypoints
                                percentage_similarity,
                            ]
                        )
                    else:
                        print(f"Unexpected match structure: {match}")  # Debug print
        print(f"Results saved successfully to {output_csv}")  # Debug print
    except Exception as e:
        print(f"Error saving results to CSV: {e}")  # Debug print


def remove_non_english_characters(filename):
    # Use regular expressions to filter out non-English characters
    return re.sub(r"[^a-zA-Z0-9._]", "", filename)


def rename_files_in_directory(directory):
    for filename in os.listdir(directory):
        # Full path of the file
        full_path = os.path.join(directory, filename)

        # Ensure only files are processed
        if os.path.isfile(full_path):
            # Remove non-English characters from the filename
            new_filename = remove_non_english_characters(filename)

            # New full path
            new_full_path = os.path.join(directory, new_filename)

            # If the new filename is different from the old one, perform the renaming operation
            if new_filename != filename:
                os.rename(full_path, new_full_path)
                print(f"Renamed: {filename} -> {new_filename}")


# Set the directory path
# directory_path = 'assets/test'

# Execute the renaming operation
# rename_files_in_directory(directory_path)
