import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
import imagehash

# python compare_images.py  --rsna-dir /data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/  --nih-dir /data/ChestX-ray14/images/  --hash-type phash  --max-hash-diff 2 --pixel-threshold 0.1  --max-workers 8 --output leakage_results.txt
def parse_args():
    """Parse command-line arguments for two-stage duplicate image detection."""
    parser = argparse.ArgumentParser(
        description="Two-stage dataset leakage detector: Perceptual Hashing (filtering) + Pixel Comparison (verification)."
    )
    parser.add_argument(
        "--rsna-dir",
        type=str,
        required=True,
        help="Path to the directory containing RSNA images.",
    )
    parser.add_argument(
        "--nih-dir",
        type=str,
        required=True,
        help="Path to the directory containing NIH images.",
    )
    parser.add_argument(
        "--hash-type",
        type=str,
        default="phash",
        choices=["phash", "ahash", "dhash", "whash"],
        help="Perceptual hash algorithm to use for Stage 1 filtering (default: phash).",
    )
    parser.add_argument(
        "--max-hash-diff",
        type=int,
        default=2,
        help="Maximum allowable Hamming distance for Stage 1 hash matching (default: 2). Set 0 for exact hash match.",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=float,
        default=0.01,
        help="Normalized pixel difference threshold for Stage 2 pixel verification (default: 0.01).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Number of threads for parallel processing (default: 8).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_leakage_matches.txt",
        help="Path to output file for saving confirmed duplicate pairs (default: data_leakage_matches.txt).",
    )
    return parser.parse_args()


def compute_single_hash(image_path, hash_type="phash"):
    """Compute the specified perceptual hash for a single image file."""
    try:
        with Image.open(image_path) as img:
            if hash_type == "phash":
                h = imagehash.phash(img)
            elif hash_type == "ahash":
                h = imagehash.average_hash(img)
            elif hash_type == "dhash":
                h = imagehash.dhash(img)
            elif hash_type == "whash":
                h = imagehash.whash(img)
            else:
                h = imagehash.phash(img)
            return image_path, h
    except Exception as e:
        print(f"Error computing hash for '{image_path}': {e}")
        return image_path, None


def compute_dataset_hashes_parallel(directory, files, hash_type="phash", max_workers=8):
    """Compute perceptual hashes for all images in a dataset using thread parallelism."""
    hashes = []
    print(f"Generating '{hash_type}' hashes for {len(files)} images in '{directory}'...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(compute_single_hash, os.path.join(directory, f), hash_type)
            for f in files
        ]
        for future in as_completed(futures):
            path, h = future.result()
            if h is not None:
                hashes.append((path, h))

    print(f"Successfully processed {len(hashes)}/{len(files)} image hashes.")
    return hashes


def find_candidate_pairs(rsna_hashes, nih_hashes, max_hash_diff=2):
    """
    Stage 1 Filtering: Screen pairs based on perceptual hash similarity.
    Returns candidate tuples: [(rsna_path, nih_path), ...].
    """
    candidates = []
    total_possible_pairs = len(rsna_hashes) * len(nih_hashes)
    print(
        f"\n[Stage 1] Screening {total_possible_pairs:,} possible pairs "
        f"(Max Hamming Distance: {max_hash_diff})..."
    )

    if max_hash_diff == 0:
        # Fast hash-table lookup O(N + M) for exact hash matches
        hash_map = defaultdict(list)
        for rsna_path, r_hash in rsna_hashes:
            hash_map[str(r_hash)].append(rsna_path)

        for nih_path, n_hash in nih_hashes:
            hash_str = str(n_hash)
            if hash_str in hash_map:
                for rsna_path in hash_map[hash_str]:
                    candidates.append((rsna_path, nih_path))
    else:
        # Hamming distance comparison between hash objects
        for rsna_path, r_hash in rsna_hashes:
            for nih_path, n_hash in nih_hashes:
                if (r_hash - n_hash) <= max_hash_diff:
                    candidates.append((rsna_path, nih_path))

    print(f"[Stage 1] Filtered down to {len(candidates)} candidate pair(s) for verification.")
    return candidates


def normalize_image(image):
    """Normalize grayscale image pixel intensities to range [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def verify_pixel_similarity(candidate_pair, threshold=0.1):
    """
    Stage 2 Verification: Perform pixel-by-pixel similarity check on candidate pair.
    Returns candidate_pair if confirmed duplicate, otherwise None.
    """
    image_path1, image_path2 = candidate_pair
    try:
        img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            return None

        if img1.shape != img2.shape:
            return None

        img1 = normalize_image(img1.astype(np.float32))
        img2 = normalize_image(img2.astype(np.float32))

        diff = np.abs(img1 - img2)

        if np.all(diff <= threshold):
            return candidate_pair

        return None
    except Exception as e:
        print(f"Error during pixel comparison for '{image_path1}' and '{image_path2}': {e}")
        return None


def verify_candidates_parallel(candidate_pairs, threshold=0.01, max_workers=8):
    """Verify candidate image pairs in parallel using pixel-level comparisons."""
    confirmed_duplicates = []
    print(f"\n[Stage 2] Verifying {len(candidate_pairs)} candidate pair(s) at pixel level...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(verify_pixel_similarity, pair, threshold)
            for pair in candidate_pairs
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                confirmed_duplicates.append(result)
                print(f"Confirmed leakage match: {result[0]} <--> {result[1]}")

    return confirmed_duplicates


def main():
    args = parse_args()

    # Path validation
    if not os.path.isdir(args.rsna_dir):
        raise FileNotFoundError(f"RSNA directory not found: {args.rsna_dir}")
    if not os.path.isdir(args.nih_dir):
        raise FileNotFoundError(f"NIH directory not found: {args.nih_dir}")

    # Read dataset file lists
    rsna_files = os.listdir(args.rsna_dir)
    nih_files = os.listdir(args.nih_dir)

    print(f"Loaded datasets: {len(rsna_files)} RSNA images | {len(nih_files)} NIH images.")

    # Step 1: Compute hashes in parallel
    rsna_hashes = compute_dataset_hashes_parallel(
        args.rsna_dir, rsna_files, hash_type=args.hash_type, max_workers=args.max_workers
    )
    nih_hashes = compute_dataset_hashes_parallel(
        args.nih_dir, nih_files, hash_type=args.hash_type, max_workers=args.max_workers
    )

    # Step 2: Stage 1 candidate filtering via perceptual hash
    candidate_pairs = find_candidate_pairs(
        rsna_hashes, nih_hashes, max_hash_diff=args.max_hash_diff
    )

    if not candidate_pairs:
        print("\nNo candidate pairs found during Stage 1 filtering. Datasets appear clean.")
        return

    # Step 3: Stage 2 pixel-level verification
    confirmed_duplicates = verify_candidates_parallel(
        candidate_pairs, threshold=args.pixel_threshold, max_workers=args.max_workers
    )

    # Step 4: Output summary and save results
    print("\n" + "=" * 60)
    if confirmed_duplicates:
        print(f"DATA LEAKAGE CONFIRMED: {len(confirmed_duplicates)} duplicate pair(s) found.")
        with open(args.output, "w", encoding="utf-8") as f:
            for rsna_path, nih_path in confirmed_duplicates:
                f.write(f"{rsna_path}\t{nih_path}\n")
                print(f" - RSNA: {rsna_path}")
                print(f"   NIH:  {nih_path}")
        print(f"\nResults successfully written to '{args.output}'.")
    else:
        print("No duplicate pairs passed Stage 2 pixel verification. Datasets appear clean.")
    print("=" * 60)


if __name__ == "__main__":
    main()