from pathlib import Path
import random

import numpy as np
import matplotlib.cm as cm
import torch
import os
import cv2


from models.matching import Matching
from models.utils import make_matching_plot, AverageTimer, read_image
from pairs_txt import generate_pairs_txt
from models.image_alignment import compare_images

torch.set_grad_enabled(False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def superglue_similarity(
    image1_path,
    image2_path,
    pairs_txt,
    output_dir,
    resize=(640, 480),
    resize_float=False,
    superglue_weights="indoor",
    max_keypoints=1024,
    keypoint_threshold=0.001,
    nms_radius=4,
    sinkhorn_iterations=20,
    match_threshold=0.6,
    viz=True,
    viz_extension="png",
    cache=False,
    shuffle=False,
    force_cpu=False,
):
    # Read image pairs
    with open(pairs_txt, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    if shuffle:
        random.Random(0).shuffle(pairs)

    # Set up device
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    print(f'Running inference on device "{device}"')

    # Configuration
    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue_weights,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(config).eval()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        matching = torch.nn.DataParallel(matching)

    matching.to(device)

    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir) / "dump_match_pairs/"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f'Will write matches to directory "{output_dir}"')

    timer = AverageTimer(newline=True)

    results = {}  # Used to store the results of all pairs

    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / f"{stem0}_{stem1}_matches.npz"
        viz_path = output_dir / f"{stem0}_{stem1}_matches.{viz_extension}"

        do_match = True
        do_viz = viz

        if cache and matches_path.exists():
            try:
                saved_results = np.load(matches_path)
                kpts0, kpts1 = saved_results["keypoints0"], saved_results["keypoints1"]
                matches, conf = (
                    saved_results["matches"],
                    saved_results["match_confidence"],
                )
                do_match = False
            except:
                raise IOError(f"Cannot load matches .npz file: {matches_path}")

        if cache and viz and viz_path.exists():
            do_viz = False

        if not (do_match or do_viz):
            timer.print(f"Finished pair {i + 1:5} of {len(pairs):5}")
            continue

        rot0, rot1 = (int(pair[2]), int(pair[3]))

        image1 = cv2.imread(Path(image1_path) / name0, 0)  # 目標影像（正確的角度）
        image2 = cv2.imread(Path(image2_path) / name1, 0)  # 待對齊的影像

        img_0, img_1 = compare_images(image1, image2)

        # Load images
        image0, inp0, scales0 = read_image(img_0, device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(img_1, device, resize, rot1, resize_float)

        if image0 is None or image1 is None:
            print(
                f"Problem reading image pair: {image1_path}/{name0}, {image2_path}/{name1}"
            )
            exit(1)

        timer.update("load_image")

        if do_match:
            # Perform matching
            pred = matching({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            timer.update("matcher")

        # Keep matching keypoints
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]

        # Save the matching results
        results[(name0, name1, rot0 * 90, len(kpts0), len(kpts1))] = len(mkpts0)

        timer.print(f"Finished pair {i + 1:5} of {len(pairs):5}")

    return results  # Return the dictionary of matching results


def superglue_similarity_image(
    image1_path,
    image2_path,
    pairs_txt,
    output_dir,
    resize=(640, 480),
    resize_float=False,
    superglue_weights="indoor",
    max_keypoints=1024,
    keypoint_threshold=0.005,
    nms_radius=4,
    sinkhorn_iterations=20,
    match_threshold=0.6,
    viz=True,
    fast_viz=False,
    show_keypoints=False,
    viz_extension="png",
    opencv_display=False,
    cache=False,
    shuffle=False,
    force_cpu=False,
):
    # Read image pairs
    with open(pairs_txt, "r") as f:
        pairs = [l.split() for l in f.readlines()]

    if shuffle:
        random.Random(0).shuffle(pairs)

    # Set up device
    device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
    print(f'Running inference on device "{device}"')

    # Configuration
    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue_weights,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }
    matching = Matching(config).eval()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        matching = torch.nn.DataParallel(matching)

    matching.to(device)

    # Create the output directory if it doesn't exist
    output_dir = Path(output_dir) / "dump_match_pairs/"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f'Will write matches to directory "{output_dir}"')

    timer = AverageTimer(newline=True)

    for i, pair in enumerate(pairs):
        rot0, rot1 = (int(pair[2]), int(pair[3]))

        name0, name1 = pair[:2]
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        matches_path = output_dir / f"{stem0}_{stem1}_matches.npz"
        viz_path = output_dir / f"{stem0}_{rot0 * 90}_{stem1}_matches.{viz_extension}"

        do_match = True
        do_viz = viz

        if cache and matches_path.exists():
            try:
                saved_results = np.load(matches_path)
                kpts0, kpts1 = saved_results["keypoints0"], saved_results["keypoints1"]
                matches, conf = (
                    saved_results["matches"],
                    saved_results["match_confidence"],
                )
                do_match = False
            except:
                raise IOError(f"Cannot load matches .npz file: {matches_path}")

        if cache and viz and viz_path.exists():
            do_viz = False

        if not (do_match or do_viz):
            timer.print(f"Finished pair {i + 1:5} of {len(pairs):5}")
            continue

        image1 = cv2.imread(Path(image1_path) / name0, 0)  # 目標影像（正確的角度）
        image2 = cv2.imread(Path(image2_path) / name1, 0)  # 待對齊的影像

        img_0, img_1 = compare_images(image1, image2)

        # Load images
        image0, inp0, scales0 = read_image(img_0, device, resize, rot0, resize_float)
        image1, inp1, scales1 = read_image(img_1, device, resize, rot1, resize_float)

        if image0 is None or image1 is None:
            print(
                f"Problem reading image pair: {image1_path}/{name0}, {image2_path}/{name1}"
            )
            exit(1)

        timer.update("load_image")

        if do_match:
            # Perform matching
            pred = matching({"image0": inp0, "image1": inp1})
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            matches, conf = pred["matches0"], pred["matching_scores0"]
            timer.update("matcher")

        # Keep matching keypoints
        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]

        # Calculate percentage similarity
        len_kpts0 = len(kpts0)
        len_kpts1 = len(kpts1)
        len_mkpts0 = len(mkpts0)
        percentage = round(
            max((len_mkpts0 / len_kpts0), (len_mkpts0 / len_kpts1)) * 100, 2
        )

        if do_viz:
            # Visualization
            color = cm.jet(mconf)
            text = [
                f"Keypoints: {len(kpts0)}:{len(kpts1)}",
                f"Matches: {len(mkpts0)}",
            ]

            if rot0 != 0 or rot1 != 0:
                text.append(f"Rotation: {rot0}:{rot1}")

            if device == "cuda":
                k_thresh = matching.module.superpoint.config["keypoint_threshold"]
                m_thresh = matching.module.superglue.config["match_threshold"]

                small_text = [
                    f"Keypoint Threshold: {k_thresh:.4f}",
                    f"Match Threshold: {m_thresh:.2f}",
                    f"Image Pair: {stem0}:{stem1}",
                ]

                make_matching_plot(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    mkpts0,
                    mkpts1,
                    color,
                    text,
                    viz_path,
                    show_keypoints,
                    fast_viz,
                    opencv_display,
                    "Matches",
                    small_text,
                )

                timer.update("viz_match")
            else:
                k_thresh = matching.superpoint.config["keypoint_threshold"]
                m_thresh = matching.superglue.config["match_threshold"]

                small_text = [
                    f"Keypoint Threshold: {k_thresh:.4f}",
                    f"Match Threshold: {m_thresh:.2f}",
                    f"Image Pair: {stem0}:{stem1}",
                ]

                make_matching_plot(
                    image0,
                    image1,
                    kpts0,
                    kpts1,
                    mkpts0,
                    mkpts1,
                    color,
                    text,
                    viz_path,
                    show_keypoints,
                    fast_viz,
                    opencv_display,
                    "Matches",
                    small_text,
                )

                timer.update("viz_match")

        timer.print(f"Finished pair {i + 1:5} of {len(pairs):5}")


def superglue_top_matches(
    input_dir, output_dir, input_image, matched_folders, datasets_folder, top_n=3
):
    # Generate the pairs.txt file
    pairs_file = generate_pairs_txt(
        input_image, matched_folders, f"datasets/{datasets_folder}"
    )

    # Use SuperGlue to perform matching and record the similarity
    superglue_results = superglue_similarity(
        input_dir, f"datasets/{datasets_folder}", pairs_file, output_dir
    )

    # Convert the dictionary to a list and sort it by similarity in descending order
    sorted_results = sorted(superglue_results.items(), key=lambda x: x[1], reverse=True)

    # Extract the top three matches
    top_matches = sorted_results[:top_n]

    # Format the results to only include the corresponding filenames and folder names, excluding extensions
    formatted_matches = []
    for (img1, img2, rotate, kpts0, kpts1), mkpts0 in top_matches:
        img1_name = Path(img1).stem  # Get the filename of img1, excluding the extension
        img2_name = Path(img2).stem  # Get the filename of img2, excluding the extension
        rotation = rotate
        folder_name = Path(
            img2
        ).parent.name  # Get the folder name where img2 is located
        formatted_matches.append(
            ((img2_name, folder_name, rotation, kpts0, kpts1), mkpts0)
        )

    return formatted_matches  # Return the formatted results
