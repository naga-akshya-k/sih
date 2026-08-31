from pathlib import Path

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from models.unet.unet_model import UNet


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "datasets" / "glas"
MODEL_PATH = PROJECT_ROOT / "outputs" / "unet" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "unet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256
THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# FIND A GlaS IMAGE + MATCHING MASK
# ============================================================

def find_test_pair():
    """
    Find a GlaS test image and its corresponding annotation.

    GlaS files commonly look like:
        testA_1.bmp
        testA_1_anno.bmp
    """

    image_candidates = []

    for pattern in ["testA_*.bmp", "testB_*.bmp", "train_*.bmp"]:
        image_candidates.extend(DATASET_DIR.rglob(pattern))

    for image_path in sorted(image_candidates):

        # Ignore annotation files
        if "_anno" in image_path.stem:
            continue

        mask_path = image_path.with_name(
            image_path.stem + "_anno.bmp"
        )

        if mask_path.exists():
            return image_path, mask_path

    raise FileNotFoundError(
        f"Could not find an image/mask pair inside {DATASET_DIR}"
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load image and convert it to a tensor suitable for U-Net.
    """

    image = Image.open(image_path).convert("RGB")

    original_size = image.size

    image_resized = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image_resized).astype(np.float32) / 255.0

    # HWC -> CHW
    image_tensor = torch.from_numpy(
        image_array.transpose(2, 0, 1)
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image, image_tensor, original_size


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):
    """
    Load the GlaS ground-truth mask.
    """

    mask = Image.open(mask_path).convert("L")

    mask_resized = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask_array = np.asarray(mask_resized)

    # Convert to binary mask
    mask_binary = (mask_array > 0).astype(np.float32)

    return mask_binary


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("Loading trained U-Net")
    print("=" * 60)

    print(f"Model:  {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found:\n{MODEL_PATH}"
        )

    model = UNet(
        in_channels=3,
        out_channels=1
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Support either a raw state_dict or a checkpoint dictionary
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("✓ Model loaded successfully")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image_tensor):

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        output = model(image_tensor)

        # Convert logits -> probability
        probability = torch.sigmoid(output)

        # Convert probability -> binary mask
        prediction = (
            probability >= THRESHOLD
        ).float()

    probability = probability.squeeze().cpu().numpy()
    prediction = prediction.squeeze().cpu().numpy()

    return probability, prediction


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    dice = (
        2.0 * intersection
        / (
            prediction.sum()
            + ground_truth.sum()
            + 1e-8
        )
    )

    return dice


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    iou = intersection / (union + 1e-8)

    return iou


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    original_image,
    ground_truth,
    probability,
    prediction,
    image_name
):

    # Save binary prediction
    prediction_image = Image.fromarray(
        (prediction * 255).astype(np.uint8)
    )

    prediction_path = (
        OUTPUT_DIR /
        f"{image_name}_prediction.png"
    )

    prediction_image.save(prediction_path)

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original_image)
    plt.title("Original H&E")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(ground_truth, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(probability, cmap="gray")
    plt.title("U-Net Probability")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(prediction, cmap="gray")
    plt.title("U-Net Prediction")
    plt.axis("off")

    plt.tight_layout()

    visualization_path = (
        OUTPUT_DIR /
        f"{image_name}_result.png"
    )

    plt.savefig(
        visualization_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    return prediction_path, visualization_path


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GlaS U-Net Prediction")
    print("=" * 60)

    # --------------------------------------------------------
    # Find image
    # --------------------------------------------------------

    image_path, mask_path = find_test_pair()

    print()
    print(f"Image : {image_path}")
    print(f"Mask  : {mask_path}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original_image, image_tensor, original_size = load_image(
        image_path
    )

    print(f"Original size: {original_size}")
    print(f"Model input : {image_tensor.shape}")

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    ground_truth = load_mask(mask_path)

    print(f"Ground truth shape: {ground_truth.shape}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print()
    print("Running U-Net prediction...")

    probability, prediction = predict(
        model,
        image_tensor
    )

    print("✓ Prediction complete")

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    dice = dice_score(
        prediction,
        ground_truth
    )

    iou = iou_score(
        prediction,
        ground_truth
    )

    print()
    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)

    print(f"Dice Score : {dice:.4f}")
    print(f"IoU Score   : {iou:.4f}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    prediction_path, visualization_path = save_results(
        original_image,
        ground_truth,
        probability,
        prediction,
        image_path.stem
    )

    print()
    print("=" * 60)
    print("RESULTS SAVED")
    print("=" * 60)

    print(f"Prediction   : {prediction_path}")
    print(f"Visualization: {visualization_path}")

    print()
    print("✓ U-Net prediction pipeline completed successfully.")


if __name__ == "__main__":
    main()