from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

from models.unet.unet_model import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_DIR = Path("datasets/glas")
OUTPUT_DIR = Path("outputs/unet")

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 1e-4

VAL_RATIO = 0.2
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# GlaS Dataset
# --------------------------------------------------

class GlasDataset(Dataset):

    def __init__(self, dataset_dir, image_size=256):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size

        # Original images only
        self.images = sorted(
            p for p in self.dataset_dir.rglob("*.bmp")
            if not p.stem.endswith("_anno")
        )

        if len(self.images) == 0:
            raise RuntimeError(
                f"No GlaS images found in {self.dataset_dir}"
            )

        # Verify masks
        for image_path in self.images:

            mask_path = image_path.with_name(
                image_path.stem + "_anno.bmp"
            )

            if not mask_path.exists():
                raise RuntimeError(
                    f"Missing mask for {image_path}"
                )

        print(f"GlaS images found: {len(self.images)}")


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):

        image_path = self.images[index]

        mask_path = image_path.with_name(
            image_path.stem + "_anno.bmp"
        )

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Load annotation
        mask = Image.open(mask_path).convert("L")

        # Resize image
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        # Resize mask using nearest-neighbour
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)

        # Convert mask to binary
        mask = (mask > 0).astype(np.float32)

        # HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# --------------------------------------------------
# Dice score
# --------------------------------------------------

def dice_score(prediction, target):

    prediction = (prediction > 0.5).float()

    intersection = (prediction * target).sum()

    dice = (
        (2.0 * intersection + 1e-7)
        /
        (prediction.sum() + target.sum() + 1e-7)
    )

    return dice.item()


# --------------------------------------------------
# Training
# --------------------------------------------------

def main():

    print("=" * 60)
    print("GlaS U-Net Training")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Dataset
    dataset = GlasDataset(
        DATASET_DIR,
        image_size=IMAGE_SIZE
    )

    # Train / validation split
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    print()
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Model
    model = UNet(
        in_channels=3,
        out_channels=1
    ).to(DEVICE)

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Output directory
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_dice = 0.0

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        model.eval()

        val_dice = 0.0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)

                probabilities = torch.sigmoid(outputs)

                val_dice += dice_score(
                    probabilities,
                    masks
                )

        val_dice /= len(val_loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {train_loss:.4f} "
            f"Val Dice: {val_dice:.4f}"
        )

        # Save best model
        if val_dice > best_dice:

            best_dice = val_dice

            model_path = OUTPUT_DIR / "best_model.pth"

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                f"  ✓ Saved best model: {model_path}"
            )


    print()
    print("=" * 60)
    print("Training complete")
    print("=" * 60)

    print(f"Best validation Dice: {best_dice:.4f}")

    print(
        f"Model saved to: "
        f"{OUTPUT_DIR / 'best_model.pth'}"
    )


if __name__ == "__main__":
    main()