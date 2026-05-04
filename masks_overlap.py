import os
from PIL import Image
import numpy as np

folder1 = "stardist_test"
folder2 = "cellpose_output_finalll/decoded_masks/combined"
output_folder = "combined_masks_final"

os.makedirs(output_folder, exist_ok=True)

# Get common image names from both folders
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))
common_files = files1.intersection(files2)

for filename in common_files:
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        path1 = os.path.join(folder1, filename)
        path2 = os.path.join(folder2, filename)

        # Open as grayscale
        img1 = Image.open(path1).convert("L")
        img2 = Image.open(path2).convert("L")

        # Make sure both images have same size
        if img1.size != img2.size:
            print(f"Skipping {filename}: size mismatch {img1.size} vs {img2.size}")
            continue

        arr1 = np.array(img1)
        arr2 = np.array(img2)

        # Convert to binary masks
        mask1 = arr1 > 0
        mask2 = arr2 > 0

        # OR operation
        combined_mask = np.logical_or(mask1, mask2)

        # Convert back to black-white image
        output = (combined_mask.astype(np.uint8) * 255)

        output_img = Image.fromarray(output)
        output_img.save(os.path.join(output_folder, filename))

        print(f"Saved: {filename}")