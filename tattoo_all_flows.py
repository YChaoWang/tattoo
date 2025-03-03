import os
import sys
import numpy as np
import pandas as pd
import cv2
import time
import itertools
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from collections import Counter
import csv

# For VGG feature extraction
from tf_keras.applications import VGG16
from tf_keras.preprocessing import image
from tf_keras.applications.vgg16 import preprocess_input
from tf_keras.models import Model

# Import required modules from your project
try:
    # Try to import from the project structure
    from vgg_compare.align.image_alignment import compare_images

    print("compare_images imported successfully.")
    from superglue_compare.match_pairs import superglue_similarity

    print("superglue_similarity imported successfully.")
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to a simpler version if needed
    # from models.image_alignment import compare_images

# Define SVM classifier weights and bias
classifier = {
    "weights_4": np.array([0.3161, -0.3982, 0.3835]),
    "bias_4": -0.1757,
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_images_in_folder(folder_path):
    """計算資料夾中的圖片數量"""
    return len(
        [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and (f.endswith(".jpg") or f.endswith(".png"))
        ]
    )


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


def extract_features(img, img_path, model, layer_names):
    """Extract VGG16 features from an image"""
    img = Image.fromarray(img).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extract features from each layer
    features = model.predict(img_array)

    # Return a dictionary with features for each layer
    return {name: features[i].flatten() for i, name in enumerate(layer_names)}


def calculate_similarity(features1, features2):
    """Calculate cosine similarity between features"""
    similarities = {}
    for layer_name in features1.keys():
        f1, f2 = features1[layer_name], features2[layer_name]
        similarity = np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))
        similarities[layer_name] = round(similarity, 3)
    return similarities


def save_similarities_to_csv(img1_path, img2_path, similarities, csv_path):
    """Save similarity results to CSV"""
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
    """Convert PNG images to JPG format"""
    if image_path.endswith(".png"):
        img = Image.open(image_path).convert("RGB")
        new_image_path = image_path.replace(".png", ".jpg")
        img.save(new_image_path, "JPEG")
        os.remove(image_path)
        return new_image_path
    return image_path


def classify_with_formula(sample, weights, bias):
    """Apply SVM formula to classify images"""
    linear_combination = np.dot(sample, weights) + bias
    return 1 if linear_combination >= 0 else 0, linear_combination


def extract_folder(path):
    """Extract the subfolder name from a path, e.g., image_1/6/2_1_r1_1.jpg -> image_1/6"""
    return "/".join(path.split("/")[:-1])


def generate_unique_pairs(image_files):
    """Generate unique pairs of images for comparison"""
    return list(itertools.combinations(image_files, 2))


def save_superglue_results_to_csv(final_results, aligned_to_original, output_csv):
    """Save SuperGlue matching results to CSV"""
    # Ensure output directory exists
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Write header if file is new
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
            # Get the original path for image2 from the mapping
            original_image2 = aligned_to_original.get(image2, image2)
            # Extract the folder from the original image2 path
            folder = extract_folder(original_image2)
            writer.writerow([image1, image2, folder, rotation, kpts1, kpts2, matches])


def main():
    start_time = time.time()
    print("Starting image comparison pipeline...")

    # Define paths
    root_dir = "image_1"
    start_folder = 1  # 要處理的資料夾起始編號
    end_folder = 1  # 要處理的資料夾結束編號
    range_start_folder = 1  # 圖片比對的資料夾範圍起始編號
    range_end_folder = 180  # 圖片比對的資料夾範圍結束編號
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    vgg_results_dir = results_dir / "vgg_comparisons"
    vgg_results_dir.mkdir(exist_ok=True)

    temp_dir = Path("temp_aligned_images")
    temp_dir.mkdir(exist_ok=True)

    superglue_results_dir = results_dir / "superglue_matches"
    superglue_results_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to map aligned images to their original paths
    aligned_to_original = {}

    # 1. PREPARE VGG MODEL
    print("Loading VGG16 model...")
    layer_names = ["block1_conv2", "block2_conv2", "block3_conv3"]
    base_model = VGG16(weights="imagenet", include_top=False)
    layer_outputs = [base_model.get_layer(name).output for name in layer_names]
    model = Model(inputs=base_model.input, outputs=layer_outputs)

    # 2. FILTER FOLDERS AND PROCESS THOSE WITH AT LEAST 2 IMAGES
    print("Filtering folders with at least 2 images...")

    # 使用新函數獲取有效資料夾
    valid_folders = get_valid_folders(root_dir, start_folder, end_folder)
    print(f"Number of valid folders to process: {len(valid_folders)}")

    # 打印找到的有效資料夾
    for folder_path in valid_folders:
        image_count = count_images_in_folder(folder_path)
        print(
            f"Folder {os.path.basename(folder_path)} has {image_count} images - included"
        )

    # 篩選比對範圍的資料夾 (這些資料夾的圖片將用於比對)
    comparison_folders = []
    for folder_num in range(range_start_folder, range_end_folder + 1):
        folder_path = os.path.join(root_dir, str(folder_num))
        if os.path.isdir(folder_path):
            image_count = count_images_in_folder(folder_path)
            if image_count > 1:  # 只要有至少 1 張圖片就可以用於比對
                comparison_folders.append(folder_num)
                print(
                    f"Comparison folder {folder_num} has {image_count} images - included for comparison"
                )

    print(f"Number of folders for comparison: {len(comparison_folders)}")

    all_comparisons = []
    total_comparisons = 0

    # 收集所有要處理的資料夾中的圖片
    process_images = []
    for folder_path in valid_folders:
        img_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        process_images.extend([convert_png_to_jpg(img_path) for img_path in img_paths])

    # 收集所有比對範圍資料夾中的圖片
    comparison_images = []
    for folder_num in comparison_folders:
        folder_path = os.path.join(root_dir, str(folder_num))
        img_paths = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(".jpg") or f.endswith(".png")
        ]
        comparison_images.extend(
            [convert_png_to_jpg(img_path) for img_path in img_paths]
        )

    print(f"Total number of images to process: {len(process_images)}")
    print(f"Total number of images for comparison: {len(comparison_images)}")

    # Process each image as the base image
    for img_idx, base_img_path in enumerate(process_images, 1):
        print(
            f"\n[{img_idx}/{len(process_images)}] Processing base image: {base_img_path}"
        )
        base_folder = extract_folder(base_img_path)
        base_img = cv2.imread(base_img_path, cv2.IMREAD_GRAYSCALE)

        if base_img is None:
            print(f"Error reading base image: {base_img_path}")
            continue

        # Compare with all images in the comparison range
        img_comparisons = []

        for compare_img_path in tqdm(
            comparison_images, desc=f"Comparing with other images"
        ):
            if base_img_path == compare_img_path:
                continue  # Skip self-comparison

            # Check if we already did this comparison in reverse
            if any(
                comp["img1"] == compare_img_path and comp["img2"] == base_img_path
                for comp in all_comparisons
            ):
                continue

            compare_img = cv2.imread(compare_img_path, cv2.IMREAD_GRAYSCALE)

            if compare_img is None:
                print(f"Error reading comparison image: {compare_img_path}")
                continue

            try:
                # Align images
                aligned_img1, aligned_img2 = compare_images(base_img, compare_img)

                # Extract features and calculate similarity
                features1 = extract_features(
                    aligned_img1, base_img_path, model, layer_names
                )
                features2 = extract_features(
                    aligned_img2, compare_img_path, model, layer_names
                )
                similarities = calculate_similarity(features1, features2)

                # Apply SVM classifier
                sample = np.array(
                    [
                        similarities["block1_conv2"],
                        similarities["block2_conv2"],
                        similarities["block3_conv3"],
                    ]
                )

                prediction, confidence = classify_with_formula(
                    sample, classifier["weights_4"], classifier["bias_4"]
                )

                comparison = {
                    "img1": base_img_path,
                    "img2": compare_img_path,
                    "prediction": prediction,
                    "confidence": confidence,
                    "folder_img1": base_folder,
                    "folder_img2": extract_folder(compare_img_path),
                }

                img_comparisons.append(comparison)
                all_comparisons.append(comparison)
                total_comparisons += 1

                # Save to intermediate CSV for the current base image
                comparison_df = pd.DataFrame([comparison])
                img_csv_path = (
                    vgg_results_dir
                    / f"comparisons_{os.path.basename(base_img_path)}.csv"
                )

                if not img_csv_path.exists():
                    comparison_df.to_csv(img_csv_path, index=False)
                else:
                    comparison_df.to_csv(
                        img_csv_path, mode="a", header=False, index=False
                    )

            except Exception as e:
                print(f"Error comparing {base_img_path} with {compare_img_path}: {e}")

        print(f"Completed {len(img_comparisons)} comparisons for {base_img_path}")

        # Calculate average confidence by subfolder for this image
        if img_comparisons:
            comparisons_df = pd.DataFrame(img_comparisons)
            # 確保只考慮至少有2張圖片的資料夾
            folders_with_counts = {}
            for folder_num in comparison_folders:
                folder_path = os.path.join(root_dir, str(folder_num))
                folders_with_counts[folder_path] = count_images_in_folder(folder_path)

            # 過濾出只有屬於有效資料夾的比較結果
            valid_comparisons_df = comparisons_df[
                comparisons_df["folder_img2"].apply(lambda x: x in folders_with_counts)
            ]

            avg_by_folder = (
                valid_comparisons_df.groupby("folder_img2")["confidence"]
                .mean()
                .reset_index()
            )

            # Sort by confidence and get top 7
            top_folders = avg_by_folder.sort_values("confidence", ascending=False).head(
                7
            )

            print(f"\nTop 7 matching folders for {base_img_path}:")
            for i, (_, row) in enumerate(top_folders.iterrows(), 1):
                folder = row["folder_img2"]
                img_count = folders_with_counts.get(folder, 0)
                print(
                    f"  {i}. {folder} - Confidence: {row['confidence']:.4f} - Images: {img_count}"
                )

            # Run SuperGlue on top 7 folders
            if len(top_folders) > 0:
                print(f"\nRunning SuperGlue with top 7 folders for {base_img_path}...")

                # Clear temp directory
                for temp_file in temp_dir.glob("*"):
                    if temp_file.is_file():
                        temp_file.unlink()

                # Clear aligned_to_original mapping for this base image
                aligned_to_original.clear()

                # Get the full color image for SuperGlue
                base_img_color = cv2.imread(base_img_path)
                if base_img_color is None:
                    print(f"Error reading color image: {base_img_path}")
                    continue

                # Save the aligned base image
                base_img_name = os.path.basename(base_img_path)
                aligned_base_path = temp_dir / f"aligned_base_{base_img_name}"
                cv2.imwrite(str(aligned_base_path), base_img_color)

                # Add base image to mapping
                aligned_to_original[str(aligned_base_path)] = base_img_path

                # Create pairs.txt for SuperGlue
                pairs_txt_path = temp_dir / "pairs.txt"
                with open(pairs_txt_path, "w") as f:
                    # Only process top 7 folders
                    for _, folder_row in top_folders.head(7).iterrows():
                        folder = folder_row["folder_img2"]
                        print(f"  Processing matches with folder: {folder}")

                        folder_path = Path(folder)
                        if not folder_path.exists():
                            print(f"  Folder does not exist: {folder}")
                            continue

                        # Get all images in this folder
                        folder_img_paths = [
                            os.path.join(folder, img_file)
                            for img_file in os.listdir(folder)
                            if img_file.endswith(".jpg") or img_file.endswith(".png")
                        ]

                        for compare_img_path in folder_img_paths:
                            compare_img = cv2.imread(compare_img_path)
                            if compare_img is None:
                                print(f"  Error reading image: {compare_img_path}")
                                continue

                            # Align images
                            aligned_base, aligned_compare = compare_images(
                                cv2.cvtColor(base_img_color, cv2.COLOR_BGR2GRAY),
                                cv2.cvtColor(compare_img, cv2.COLOR_BGR2GRAY),
                            )

                            # Convert back to color for SuperGlue
                            aligned_base_color = cv2.cvtColor(
                                aligned_base, cv2.COLOR_GRAY2BGR
                            )
                            aligned_compare_color = cv2.cvtColor(
                                aligned_compare, cv2.COLOR_GRAY2BGR
                            )

                            # Save aligned comparison image
                            compare_img_name = os.path.basename(compare_img_path)
                            aligned_compare_path = (
                                temp_dir / f"aligned_compare_{compare_img_name}"
                            )
                            cv2.imwrite(str(aligned_base_path), aligned_base_color)
                            cv2.imwrite(
                                str(aligned_compare_path), aligned_compare_color
                            )

                            # Store the mapping from aligned image to original path
                            aligned_to_original[str(aligned_compare_path)] = (
                                compare_img_path
                            )

                            # Add to pairs.txt
                            f.write(
                                f"{aligned_base_path.name} {aligned_compare_path.name} 0 0\n"
                            )

                # Check if pairs.txt has content
                if pairs_txt_path.stat().st_size == 0:
                    print(f"  No valid pairs generated for {base_img_path}")
                    continue

                # Create output directory for SuperGlue results
                base_folder_name = os.path.basename(base_folder)
                superglue_output_dir = (
                    superglue_results_dir
                    / f"base_{base_folder_name}_{os.path.basename(base_img_path)}"
                )
                superglue_output_dir.mkdir(parents=True, exist_ok=True)

                # Run SuperGlue
                print(f"  Running SuperGlue for {base_img_path}...")
                try:
                    results = superglue_similarity(
                        str(temp_dir),
                        str(temp_dir),
                        str(pairs_txt_path),
                        str(superglue_output_dir),
                    )

                    # Save SuperGlue results with original path mapping
                    save_superglue_results_to_csv(
                        results,
                        aligned_to_original,
                        str(superglue_results_dir / "all_tatoos_superglue_matches.csv"),
                    )

                    print(f"  SuperGlue matching completed for {base_img_path}")
                except Exception as e:
                    print(f"  Error running SuperGlue: {e}")
            else:
                print(f"No matching folders found for {base_img_path}")

        # Save progress after each image
        progress_info = {
            "completed_images": img_idx,
            "total_images": len(process_images),
            "total_comparisons": total_comparisons,
            "last_processed": base_img_path,
            "elapsed_time": time.time() - start_time,
        }

        pd.DataFrame([progress_info]).to_csv(results_dir / "progress.csv", index=False)

        # Print progress summary
        elapsed = time.time() - start_time
        images_left = len(process_images) - img_idx
        avg_time_per_img = elapsed / img_idx if img_idx > 0 else 0
        est_time_left = avg_time_per_img * images_left

        print(
            f"\nProgress: {img_idx}/{len(process_images)} images processed ({img_idx/len(process_images)*100:.1f}%)"
        )
        print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(
            f"Estimated time remaining: {est_time_left:.1f} seconds ({est_time_left/60:.1f} minutes)"
        )

    # Save all VGG comparisons to a single CSV at the end
    print("\nSaving all VGG comparison results...")
    all_comparisons_df = pd.DataFrame(all_comparisons)
    all_comparisons_file = results_dir / "all_tattoos_vgg_comparisons.csv"
    all_comparisons_df.to_csv(all_comparisons_file, index=False)

    # Save top 7 folders for all images
    print("Saving top 7 confidence subfolders for all images...")

    # 確保只考慮比對範圍內的資料夾
    folders_with_counts = {}
    for folder_num in comparison_folders:
        folder_path = os.path.join(root_dir, str(folder_num))
        folders_with_counts[folder_path] = count_images_in_folder(folder_path)

    result_data = []
    for img_path in all_comparisons_df["img1"].unique():
        # Get all comparisons for this image
        img_df = all_comparisons_df[all_comparisons_df["img1"] == img_path]

        # 過濾只保留有效資料夾的比較結果
        img_df = img_df[img_df["folder_img2"].apply(lambda x: x in folders_with_counts)]

        # Calculate average confidence by subfolder
        avg_by_folder = img_df.groupby("folder_img2")["confidence"].mean().reset_index()

        # Sort by confidence and get top 7
        top_folders = avg_by_folder.sort_values("confidence", ascending=False).head(7)

        row_data = {"img1": img_path}

        # Add top 7 folders and confidences
        for i, (_, row) in enumerate(top_folders.iterrows(), 1):
            row_data[f"folder_img2_no{i}"] = row["folder_img2"]
            row_data[f"confidence_{i}"] = row["confidence"]

        result_data.append(row_data)

    # Create and save top folders DataFrame
    if result_data:
        result_df = pd.DataFrame(result_data)

        # Ensure all columns exist
        columns = ["img1"]
        for i in range(1, 8):
            columns.extend([f"folder_img2_no{i}", f"confidence_{i}"])

        for col in columns:
            if col not in result_df.columns:
                result_df[col] = None

        # Reorder columns and rename as needed
        result_df = result_df[columns]
        new_columns = ["img1"]
        for i in range(1, 8):
            new_columns.extend([f"folder_img2_no{i}", "confidence"])

        result_df.columns = new_columns
        top_folders_file = results_dir / "top_confidence_folders.csv"
        result_df.to_csv(top_folders_file, index=False)

    total_time = time.time() - start_time
    print(
        f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    print(f"Total comparisons: {total_comparisons}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
