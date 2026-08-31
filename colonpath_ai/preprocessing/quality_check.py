from pathlib import Path
import cv2
import numpy as np

# Find image
project_root = Path(__file__).resolve().parent.parent

matches = list(
    (project_root / "datasets" / "glas").rglob("testA_1.bmp")
)

if not matches:
    print("ERROR: Image not found")
    exit()

image_path = matches[0]

image = cv2.imread(str(image_path))

if image is None:
    print("ERROR: Could not read image")
    exit()

# -----------------------------
# 1. Resolution
# -----------------------------

height, width = image.shape[:2]

print("IMAGE QUALITY CHECK")
print("===================")

print(f"Resolution: {width} x {height}")

# -----------------------------
# 2. Blur detection
# -----------------------------

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

laplacian_variance = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

print(f"Laplacian variance: {laplacian_variance:.2f}")

# -----------------------------
# 3. Brightness
# -----------------------------

brightness = np.mean(gray)

print(f"Mean brightness: {brightness:.2f}")

# -----------------------------
# 4. Contrast
# -----------------------------

contrast = np.std(gray)

print(f"Contrast (std): {contrast:.2f}")

# -----------------------------
# 5. Saturation
# -----------------------------

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

mean_saturation = np.mean(hsv[:, :, 1])

print(f"Mean saturation: {mean_saturation:.2f}")

# -----------------------------
# Overall result
# -----------------------------

print("\nQUALITY RESULT")

if laplacian_variance < 50:
    print("Blur: HIGH")
else:
    print("Blur: ACCEPTABLE")

if brightness < 50:
    print("Brightness: TOO DARK")
elif brightness > 220:
    print("Brightness: VERY BRIGHT")
else:
    print("Brightness: ACCEPTABLE")

if contrast < 20:
    print("Contrast: LOW")
else:
    print("Contrast: ACCEPTABLE")