import os
from PIL import Image
import numpy as np

actual_folder = "cellpose_output/decoded_masks/combined"
# predicted_folder = "cellpose_output_finalll/decoded_masks/combined"
predicted_folder = "combined_masks_final"
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

pred_files = set(
    f for f in os.listdir(predicted_folder)
    if f.lower().endswith(image_extensions)
)

actual_files = set(
    f for f in os.listdir(actual_folder)
    if f.lower().endswith(image_extensions)
)

common_files = sorted(pred_files.intersection(actual_files))

if not common_files:
    raise ValueError("No matching filenames found.")

iou_scores = []

for filename in common_files:
    pred_path = os.path.join(predicted_folder, filename)
    actual_path = os.path.join(actual_folder, filename)

    pred_img = Image.open(pred_path).convert("L")
    actual_img = Image.open(actual_path).convert("L")

    if pred_img.size != actual_img.size:
        print(f"Skipping {filename}: size mismatch")
        continue

    pred_mask = np.array(pred_img) > 0
    actual_mask = np.array(actual_img) > 0

    intersection = np.logical_and(pred_mask, actual_mask).sum()
    union = np.logical_or(pred_mask, actual_mask).sum()

    if union == 0:
        iou = 1.0
    else:
        iou = intersection / union

    iou_scores.append(iou)

average_iou = np.mean(iou_scores)

print("Average IoU:", average_iou)