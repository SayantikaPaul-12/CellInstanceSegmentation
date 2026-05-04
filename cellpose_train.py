"""
=============================================================================
Cellpose Fine-Tuning Pipeline for Kaggle Sartorius Cell Instance Segmentation
=============================================================================

This script provides an end-to-end pipeline:
  1. Install dependencies
  2. Load & parse competition data (images + RLE masks)
  3. Convert RLE annotations → instance label masks
  4. Fine-tune Cellpose on competition data
  5. Run inference with tuned model
  6. Generate submission CSV

Usage:
  - Run cells sequentially in a Kaggle notebook or Jupyter environment.
  - Set COMPETITION_DIR to your data path.

Requirements:
  pip install cellpose opencv-python-headless pandas numpy tqdm scikit-image
"""

# ============================================================================
# 0. CONFIGURATION
# ============================================================================

import os

# -- PATHS (adjust these to your environment) --------------------------------
TRAIN_CSV       = "data/sartorius-cell-instance-segmentation/mask_elong_final.xlsx"
TRAIN_IMG_DIR   = "data/sartorius-cell-instance-segmentation/train"
TEST_IMG_DIR    = "data/sartorius-cell-instance-segmentation/test"

OUTPUT_DIR      = "./cellpose_output_elong_final_1"
MODEL_DIR       = os.path.join(OUTPUT_DIR, "models")
MASKS_DIR       = os.path.join(OUTPUT_DIR, "train_masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

# -- TRAINING SETTINGS -------------------------------------------------------
CELL_TYPES       = ["shsy5y"]   # 3 cell types in competition , "astro", "cort",
PRETRAINED_MODEL = "cyto2"                         # base model to fine-tune
N_EPOCHS         = 200                             # fine-tuning epochs
LEARNING_RATE    = 1e-5                            # Cellpose v4 default LR
BATCH_SIZE       = 1                               # keep low for large images on 24GB GPU
USE_GPU          = True

# -- INFERENCE SETTINGS -------------------------------------------------------
FLOW_THRESHOLD   = 0.4     # increase to get fewer masks (default 0.4)
CELLPROB_THRESH  = 0.0     # decrease to get more masks  (default 0.0)
MIN_SIZE         = 15       # minimum mask size in pixels


# ============================================================================
# 1. IMPORTS
# ============================================================================

import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import cv2
from skimage.io import imread

# Cellpose imports
from cellpose import models, io, train


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def rle_decode_sartorius(rle_str: str, height: int, width: int) -> np.ndarray:
    """
    Decode Sartorius RLE: 'start length start length ...'
    - starts are 1-based
    - flattening is column-major

    Returns (H, W) uint8 mask (0/1).
    """
    if pd.isna(rle_str) or rle_str == "":
        return np.zeros((height, width), dtype=np.uint8)
    s = np.asarray(rle_str.split(), dtype=np.int64)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    flat = np.zeros(height * width, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        flat[lo:hi] = 1
    return flat.reshape((width, height), order="F").T


def normalize_id(x):
    """
    Normalize image IDs that may have been mangled into scientific
    notation (e.g. '1.23456E+17') back to integer strings.
    """
    if isinstance(x, str) and re.match(r"^\d+\.\d+E\+\d+$", x):
        return str(int(float(x)))
    return str(x)


def rle_encode(mask: np.ndarray) -> str:
    """
    Encode a binary mask into RLE string for submission.

    Args:
        mask: Binary mask of shape (height, width)

    Returns:
        RLE encoded string
    """
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def build_instance_masks(df: pd.DataFrame, image_id: str, shape: tuple) -> np.ndarray:
    """
    Build an instance segmentation label image from RLE annotations.
    Each instance gets a unique integer label (1, 2, 3, ...).

    Args:
        df: Training DataFrame with columns [id, annotation, ...]
        image_id: The image ID to build masks for
        shape: (height, width) of the image

    Returns:
        Label mask of shape (height, width), dtype int32
    """
    norm_id = normalize_id(image_id)
    rows = df[df["id"] == norm_id]
    h, w = shape
    label_mask = np.zeros(shape, dtype=np.int32)
    for idx, (_, row) in enumerate(rows.iterrows(), start=1):
        binary = rle_decode_sartorius(row["annotation"], h, w)
        label_mask[binary > 0] = idx
    return label_mask


# ============================================================================
# 3. PREPARE TRAINING DATA
# ============================================================================

def prepare_training_data(
    train_csv: str,
    train_img_dir: str,
    masks_dir: str,
    max_images_per_type: int = None,
):
    """
    Convert competition annotations to Cellpose-compatible format.

    Cellpose expects:
      - images as .png/.tif
      - label masks as _masks.png/.tif (same base name)

    Returns:
        Dictionary mapping cell_type -> list of (image_path, mask_path)
    """
    print("Loading training CSV...")
    df = pd.read_excel(train_csv)

    # Normalize IDs that may have been mangled into scientific notation
    df["id"] = df["id"].apply(normalize_id)

    # Extract cell type from the 'cell_type' column
    print(f"Total annotations: {len(df)}")
    print(f"Unique images:     {df['id'].nunique()}")
    print(f"Cell types:        {df['cell_type'].unique()}")

    data_by_type = {ct: [] for ct in CELL_TYPES}

    for cell_type in CELL_TYPES:
        ct_dir = os.path.join(masks_dir, cell_type)
        os.makedirs(ct_dir, exist_ok=True)

        ct_df = df[df["cell_type"] == cell_type]
        image_ids = ct_df["id"].unique()

        if max_images_per_type:
            image_ids = image_ids[:max_images_per_type]

        print(f"\nProcessing {cell_type}: {len(image_ids)} images")

        for img_id in tqdm(image_ids, desc=f"  {cell_type}"):
            # Read image
            img_path = os.path.join(train_img_dir, f"{img_id}.png")
            if not os.path.exists(img_path):
                continue

            img = imread(img_path)
            h, w = img.shape[:2]

            # Build instance label mask
            label_mask = build_instance_masks(ct_df, img_id, (h, w))

            # Save image and mask in Cellpose naming convention
            out_img_path  = os.path.join(ct_dir, f"{img_id}.png")
            out_mask_path = os.path.join(ct_dir, f"{img_id}_masks.png")

            # Save as 16-bit to support > 255 instances
            cv2.imwrite(out_img_path, img)
            cv2.imwrite(out_mask_path, label_mask.astype(np.uint16))

            data_by_type[cell_type].append((out_img_path, out_mask_path))

    for ct, items in data_by_type.items():
        print(f"  {ct}: {len(items)} image-mask pairs saved")

    return data_by_type


# ============================================================================
# 4. FINE-TUNE CELLPOSE
# ============================================================================

def finetune_cellpose(
    cell_type: str,
    train_dir: str,
    model_name: str = PRETRAINED_MODEL,
    n_epochs: int = N_EPOCHS,
    learning_rate: float = LEARNING_RATE,
):
    """
    Fine-tune a Cellpose model on a specific cell type.

    Strategy: Fine-tune from a pretrained model (cyto2) which already
    understands cell-like structures. This requires far fewer epochs
    than training from scratch.

    Args:
        cell_type: One of "astro", "cort", "shsy5y"
        train_dir: Directory containing images + _masks.png files
        model_name: Pretrained model to start from
        n_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate

    Returns:
        Path to the saved fine-tuned model
    """
    print(f"\n{'='*60}")
    print(f"Fine-tuning Cellpose for: {cell_type}")
    print(f"  Base model:     {model_name}")
    print(f"  Epochs:         {n_epochs}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Training dir:   {train_dir}")
    print(f"{'='*60}\n")

    # Load images and masks using Cellpose I/O
    output = io.load_train_test_data(train_dir, mask_filter="_masks")
    images, labels, image_names = output[:3]

    # If there's test data it will be in output[3:], but we handle that
    # via validation split below
    test_images = output[3] if len(output) > 3 else None
    test_labels = output[4] if len(output) > 4 else None

    print(f"  Loaded {len(images)} training images")

    # Initialize model from pretrained weights
    # Use pretrained_model= to load the smaller cyto2 model explicitly.
    # The default in v4 is "cpsam" (Cellpose-SAM transformer) which is
    # much larger and requires more GPU memory.
    model = models.CellposeModel(
        gpu=USE_GPU,
        pretrained_model=model_name,
    )

    # Fine-tune
    save_path = os.path.join(MODEL_DIR, cell_type)
    os.makedirs(save_path, exist_ok=True)

    new_model_path, train_losses, test_losses = train.train_seg(
        model.net,
        train_data=images,
        train_labels=labels,
        test_data=test_images,
        test_labels=test_labels,
        save_path=save_path,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        min_train_masks=1,
        model_name=f"cellpose_{cell_type}",
    )

    print(f"\n  Model saved to: {new_model_path}")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    if test_losses is not None and len(test_losses) > 0:
        print(f"  Final test loss:  {test_losses[-1]:.4f}")
    return new_model_path


# ============================================================================
# 5. INFERENCE
# ============================================================================

def run_inference(
    model_paths: dict,
    test_img_dir: str,
    cell_type_map: dict = None,
):
    """
    Run inference on test images.

    For the Sartorius competition, each test image has a known cell type.
    If cell_type_map is provided, we use the type-specific model.
    Otherwise we use a single model for all images.

    Args:
        model_paths: dict of {cell_type: model_path}
        test_img_dir: Directory containing test images
        cell_type_map: Optional dict of {image_id: cell_type}

    Returns:
        List of dicts with keys: id, predicted (RLE encoded masks)
    """
    print("\nRunning inference on test images...")

    # Load models
    loaded_models = {}
    for ct, path in model_paths.items():
        loaded_models[ct] = models.CellposeModel(
            gpu=USE_GPU,
            pretrained_model=path,
        )
        print(f"  Loaded model for {ct}: {path}")

    # If no cell type map, use a default model
    default_ct = list(model_paths.keys())[0]

    results = []
    test_images = sorted(Path(test_img_dir).glob("*.png"))
    print(f"  Found {len(test_images)} test images\n")

    for img_path in tqdm(test_images, desc="  Inference"):
        img_id = img_path.stem
        img = imread(str(img_path))

        # Select model based on cell type
        ct = cell_type_map.get(img_id, default_ct) if cell_type_map else default_ct
        model = loaded_models.get(ct, loaded_models[default_ct])

        # Run Cellpose
        masks, flows, styles = model.eval(
            img,
            flow_threshold=FLOW_THRESHOLD,
            cellprob_threshold=CELLPROB_THRESH,
            min_size=MIN_SIZE,
        )

        # Convert each instance mask to RLE
        n_instances = masks.max()
        for i in range(1, n_instances + 1):
            binary = (masks == i).astype(np.uint8)
            rle = rle_encode(binary)
            results.append({
                "id": img_id,
                "predicted": rle,
            })

    print(f"\n  Total predictions: {len(results)}")
    return results


# ============================================================================
# 6. GENERATE SUBMISSION
# ============================================================================

def create_submission(results: list, output_path: str):
    """
    Create a Kaggle submission CSV.

    Args:
        results: List of dicts with 'id' and 'predicted' keys
        output_path: Path to save submission.csv
    """
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(f"  Shape: {sub_df.shape}")
    print(f"  Unique images: {sub_df['id'].nunique()}")
    print(f"  Preview:\n{sub_df.head()}")


# ============================================================================
# 7. EVALUATION (local validation)
# ============================================================================

def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / max(union, 1)


def compute_map_score(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    thresholds: np.ndarray = np.arange(0.5, 1.0, 0.05),
) -> float:
    """
    Compute mean Average Precision (mAP) at IoU thresholds [0.5, 0.55, ..., 0.95].
    This is the competition metric.

    Args:
        pred_masks: Predicted instance label mask
        gt_masks: Ground truth instance label mask
        thresholds: IoU thresholds

    Returns:
        mAP score
    """
    pred_ids = [i for i in np.unique(pred_masks) if i > 0]
    gt_ids   = [i for i in np.unique(gt_masks) if i > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    # Compute IoU matrix
    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            iou_matrix[i, j] = compute_iou(pred_masks == pid, gt_masks == gid)

    scores = []
    for thresh in thresholds:
        matched_gt = set()
        matched_pred = set()

        # Greedy matching: best IoU first
        sorted_pairs = np.dstack(np.unravel_index(
            np.argsort(-iou_matrix, axis=None), iou_matrix.shape
        ))[0]

        tp = 0
        for pi, gi in sorted_pairs:
            if pi in matched_pred or gi in matched_gt:
                continue
            if iou_matrix[pi, gi] >= thresh:
                tp += 1
                matched_pred.add(pi)
                matched_gt.add(gi)

        fp = len(pred_ids) - tp
        fn = len(gt_ids) - tp
        precision = tp / max(tp + fp + fn, 1)
        scores.append(precision)

    return np.mean(scores)


# ============================================================================
# 8. MAIN PIPELINE
# ============================================================================

def main():
    """
    Full pipeline: data prep → fine-tune → inference → submission.
    """

    # ---- Step 1: Prepare training data ------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Preparing training data")
    print("=" * 70)

    data_by_type = prepare_training_data(
        train_csv=TRAIN_CSV,
        train_img_dir=TRAIN_IMG_DIR,
        masks_dir=MASKS_DIR,
        max_images_per_type=None,  # Set to e.g. 50 for quick testing
    )

    # ---- Step 2: Fine-tune per cell type ----------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Fine-tuning Cellpose models")
    print("=" * 70)

    model_paths = {}
    for cell_type in CELL_TYPES:
        train_dir = os.path.join(MASKS_DIR, cell_type)
        model_path = finetune_cellpose(
            cell_type=cell_type,
            train_dir=train_dir,
            n_epochs=N_EPOCHS,
        )
        model_paths[cell_type] = model_path

    # ---- Step 3: Run inference --------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3: Running inference on test set")
    print("=" * 70)

    # NOTE: In the real competition, you'd determine cell type from
    # metadata or a classifier. For simplicity, we use a single model
    # or you can build a cell-type classifier as a preprocessing step.
    results = run_inference(
        model_paths=model_paths,
        test_img_dir=TEST_IMG_DIR,
        cell_type_map=None,  # Provide if you have a classifier
    )

    # ---- Step 4: Create submission ----------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Creating submission")
    print("=" * 70)

    submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
    create_submission(results, submission_path)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)


# ============================================================================
# 9. QUICK-START EXAMPLES
# ============================================================================

def quick_test_single_image():
    """
    Quick test: run pretrained Cellpose on a single image.
    Good for verifying your setup works before fine-tuning.
    """
    # Find a test image
    test_imgs = sorted(Path(TRAIN_IMG_DIR).glob("*.png"))
    if not test_imgs:
        print("No images found. Check TRAIN_IMG_DIR path.")
        return

    img = imread(str(test_imgs[0]))
    print(f"Image shape: {img.shape}")

    # Run pretrained model (no fine-tuning)
    model = models.CellposeModel(gpu=USE_GPU)
    masks, flows, styles = model.eval(
        img,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )

    print(f"Detected {masks.max()} instances")
    return masks


def optimize_thresholds(model, val_images, val_labels):
    """
    Grid search over flow_threshold and cellprob_threshold
    to maximize mAP on a validation set.

    Args:
        model: Trained CellposeModel
        val_images: List of validation images
        val_labels: List of ground truth label masks

    Returns:
        Best (flow_threshold, cellprob_threshold, mAP)
    """
    best_score = 0
    best_params = (0.4, 0.0)

    flow_thresholds = np.arange(0.1, 0.9, 0.1)
    cellprob_thresholds = np.arange(-2.0, 2.0, 0.5)

    for ft in flow_thresholds:
        for cp in cellprob_thresholds:
            scores = []
            for img, gt in zip(val_images, val_labels):
                masks, _, _ = model.eval(
                    img,
                    flow_threshold=ft,
                    cellprob_threshold=cp,
                )
                score = compute_map_score(masks, gt)
                scores.append(score)

            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = (ft, cp)
                print(f"  New best: flow={ft:.1f}, cellprob={cp:.1f}, mAP={mean_score:.4f}")

    print(f"\nBest params: flow={best_params[0]:.1f}, cellprob={best_params[1]:.1f}")
    print(f"Best mAP:    {best_score:.4f}")
    return best_params[0], best_params[1], best_score


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()