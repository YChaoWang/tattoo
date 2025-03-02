from pathlib import Path
import os


def generate_pairs_txt_mode(main_folder, mode="single"):
    """
    Generate a pairs.txt file for image matching and return the file path.

    Parameters:
    - main_folder: The main folder containing images or subfolders.
    - mode: 'single' to generate pairs.txt for images in the main folder,
            'subfolders' to generate pairs.txt for each subfolder.

    Returns:
    - The file path(s) of the generated pairs.txt file(s).
    """

    input_dir = Path(main_folder)
    txt_dir = Path(
        "superglue_compare/txt"
    )  # Place the txt folder at the same level as main_folder
    txt_dir.mkdir(
        parents=True, exist_ok=True
    )  # Create the 'txt' directory if it doesn't exist
    generated_files = []

    if mode == "single":
        # Get all png or jpg files in the main folder (including all subfolders)
        png_files = [
            f.name for f in input_dir.rglob("*") if f.suffix in [".png", ".jpg"]
        ]
        output_file = txt_dir / "pairs.txt"

        # Generate the pairs.txt file
        with open(output_file, "w") as f:
            for i in range(len(png_files)):
                for j in range(i + 1, len(png_files)):
                    f.write(f"{png_files[i]} {png_files[j]} 0 0\n")
                    f.write(f"{png_files[i]} {png_files[j]} 1 0\n")
                    f.write(f"{png_files[i]} {png_files[j]} 2 0\n")
                    f.write(f"{png_files[i]} {png_files[j]} 3 0\n")

        print(f"pairs.txt file has been generated at: {output_file}")
        generated_files.append(output_file)

    elif mode == "subfolders":
        # Get all subfolders in the main folder
        subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

        # Iterate through each subfolder
        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)
            output_file = txt_dir / f"{subfolder_name}.txt"

            # Get all png or jpg files in the subfolder
            png_files = [
                f
                for f in os.listdir(subfolder)
                if f.endswith(".png") or f.endswith(".jpg")
            ]

            # Generate the pairs.txt file
            with open(output_file, "w") as f:
                for i in range(len(png_files)):
                    for j in range(i + 1, len(png_files)):
                        f.write(f"{png_files[i]} {png_files[j]} 0 0\n")
                        f.write(f"{png_files[i]} {png_files[j]} 1 0\n")
                        f.write(f"{png_files[i]} {png_files[j]} 2 0\n")
                        f.write(f"{png_files[i]} {png_files[j]} 3 0\n")

            print(f"{subfolder_name}.txt file has been generated at: {output_file}")
            generated_files.append(output_file)

    else:
        raise ValueError(
            "Invalid mode. Use 'single' for main folder or 'subfolders' for each subfolder."
        )

    return generated_files


# Example usage:
# files = generate_pairs_txt('path/to/main_folder', mode='single')  # Process a single folder
# print(files)
# files = generate_pairs_txt('path/to/main_folder', mode='subfolders')  # Process all subfolders
# print(files)


def generate_pairs_txt_no_rotate(input_image, matched_folders, datasets_dir):
    pairs_file = "txt/pairs.txt"
    with open(pairs_file, "w") as file:
        for folder in matched_folders:
            folder_path = os.path.join(datasets_dir, folder)
            for dataset_image in os.listdir(folder_path):
                if dataset_image.endswith(".png"):
                    # Write each pair of images to pairs.txt in the format "image1 image2 0 0"
                    file.write(
                        f"{input_image} {os.path.join(folder, dataset_image)} 0 0\n"
                    )
    return pairs_file


def generate_pairs_txt(input_image, matched_folders, datasets_dir):
    pairs_file = "tools/superglue/txt/pairs.txt"
    with open(pairs_file, "w") as file:
        for folder in matched_folders:
            folder_path = os.path.join(datasets_dir, folder)
            for dataset_image in os.listdir(folder_path):
                if dataset_image.endswith(".png"):
                    # Write each pair of images to pairs.txt in the format "image1 image2 0 0"
                    file.write(
                        f"{input_image} {os.path.join(folder, dataset_image)} 0 0\n"
                    )
                    file.write(
                        f"{input_image} {os.path.join(folder, dataset_image)} 1 0\n"
                    )
                    file.write(
                        f"{input_image} {os.path.join(folder, dataset_image)} 2 0\n"
                    )
                    file.write(
                        f"{input_image} {os.path.join(folder, dataset_image)} 3 0\n"
                    )
    return pairs_file


def generate_pairs_txt_final(final_results):
    pairs_file = "tools/superglue/txt/pairs.txt"

    with open(pairs_file, "w") as file:
        for main_image, pairs_image in final_results.items():
            for image in pairs_image:
                folder_name = image[0][1]
                image_name = image[0][0]
                rotation = image[0][2]
                image_file = f"{image_name}.png"
                file.write(
                    f"{main_image} {os.path.join(folder_name, image_file)} {int(rotation/90)} 0\n"
                )
    return pairs_file
