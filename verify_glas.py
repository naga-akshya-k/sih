from pathlib import Path
from PIL import Image

# GlaS dataset location
DATASET_DIR = Path("datasets/glas")

# Find original images only
images = sorted(
    p for p in DATASET_DIR.rglob("*.bmp")
    if not p.stem.endswith("_anno")
)

print("=" * 60)
print("GlaS Dataset Verification")
print("=" * 60)

print(f"Dataset directory : {DATASET_DIR}")
print(f"Original images   : {len(images)}")

missing_masks = []
bad_dimensions = []
unreadable = []
empty_masks = []

for image_path in images:
    # Example:
    # testA_10.bmp -> testA_10_anno.bmp
    mask_path = image_path.with_name(
        image_path.stem + "_anno.bmp"
    )

    # Check mask exists
    if not mask_path.exists():
        missing_masks.append(image_path)
        continue

    try:
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Check dimensions
        if image.size != mask.size:
            bad_dimensions.append(
                (image_path, image.size, mask.size)
            )

        # Check that mask has non-zero pixels
        mask_values = mask.convert("L").getextrema()

        if mask_values[1] == 0:
            empty_masks.append(mask_path)

    except Exception as e:
        unreadable.append((image_path, str(e)))


print()
print("-" * 60)
print("RESULTS")
print("-" * 60)

print(f"Images found       : {len(images)}")
print(f"Missing masks      : {len(missing_masks)}")
print(f"Dimension errors   : {len(bad_dimensions)}")
print(f"Unreadable files   : {len(unreadable)}")
print(f"Empty masks        : {len(empty_masks)}")

print()

if not missing_masks and not bad_dimensions and not unreadable:
    print("SUCCESS: GlaS image/mask pairs are valid.")
else:
    print("WARNING: Some dataset problems were detected.")

print("=" * 60)