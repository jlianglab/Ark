import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def normalize_image(image):
    """Normalize grayscale image pixel intensities to range [0, 1]."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)
    
def are_images_similar(image_path1, image_path2, threshold=0.1):
    """
    Compare two images pixel-by-pixel after normalization.
    Returns the file paths tuple if similar, otherwise None.
    """
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
            return (image_path1, image_path2)
        return None
    except Exception as e:
        print(f"Error comparing '{image_path1}' and '{image_path2}': {e}")
        return None

def compare_images_parallel(covid_dir, covid_files, nih_dir, nih_files, threshold=0.1, max_workers=8):
    """Parallel pairwise image comparison across datasets."""
    duplicates = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, covid_file in enumerate(covid_files):
            print(f"Processing image {i + 1}/{len(covid_files)}: {covid_file}")
            covid_path = os.path.join(covid_dir, covid_file)
            
            futures = [
                executor.submit(
                    are_images_similar, 
                    covid_path, 
                    os.path.join(nih_dir, nih_file), 
                    threshold
                )
                for nih_file in nih_files
            ]

            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        covid_path_match, nih_path_match = result
                        duplicates.append((covid_path_match, nih_path_match))
                        print(f"Found duplicate: {covid_path_match} <--> {nih_path_match}")
                except Exception as e:
                    print(f"Task execution error: {e}")

    return duplicates

# File paths
covid_test_dir = "covidx-cxr3/test/"
nih_train_dir = "ChestX-ray14/images/"
output_txt_file = "covidx_nih_duplicates.txt"

covid_files = os.listdir(covid_test_dir)

with open("Xray14_train_official.txt", 'r') as file:
    nih_files = [line.strip().split()[0] for line in file if line.strip()]

# Parallel comparison
duplicates = compare_images_parallel(
    covid_test_dir, 
    covid_files, 
    nih_train_dir, 
    nih_files, 
    threshold=0.1, 
    max_workers=16
)

# Output summary and write to file
if duplicates:
    print(f"\nData leakage detected! Found {len(duplicates)} duplicate pair(s):")
    with open(output_txt_file, 'w', encoding='utf-8') as f:
        for covid_path, nih_path in duplicates:
            print(f"COVIDx-3 path: {covid_path}")
            print(f"NIH path:     {nih_path}")
            # Save tab-separated relative/full paths or filenames
            f.write(f"{covid_path}\t{nih_path}\n")
            
    print(f"\nSuccessfully saved duplicate pairs to '{output_txt_file}'.")
else:
    print("\nNo duplicate images found.")