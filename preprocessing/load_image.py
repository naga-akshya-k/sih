from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Project root
project_root = Path(__file__).resolve().parent.parent

# Find image and annotation
image_files = list(
    (project_root / "datasets" / "glas").rglob("testA_1.bmp")
)

anno_files = list(
    (project_root / "datasets" / "glas").rglob("testA_1_anno.bmp")
)

if not image_files:
    print("ERROR: H&E image not found")
    exit()

if not anno_files:
    print("ERROR: Annotation not found")
    exit()

image_path = image_files[0]
anno_path = anno_files[0]

print("Image:", image_path)
print("Annotation:", anno_path)

# Load H&E image
image = cv2.imread(str(image_path))

# Load annotation
annotation = cv2.imread(str(anno_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    print("ERROR: Could not load H&E image")
    exit()

if annotation is None:
    print("ERROR: Could not load annotation")
    exit()

print("\nImage loaded successfully!")
print("Image shape:", image.shape)

print("\nAnnotation loaded successfully!")
print("Annotation shape:", annotation.shape)

# Convert image for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display both
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original H&E Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(annotation, cmap="gray")
plt.title("Gland Ground-Truth Annotation")
plt.axis("off")

plt.tight_layout()
plt.show()