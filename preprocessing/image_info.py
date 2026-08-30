from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent.parent

image_files = list(
    (project_root / "datasets" / "glas").rglob("testA_1.bmp")
)

image_path = image_files[0]

image = cv2.imread(str(image_path))

print("Image information")
print("-----------------")

print("File:", image_path)
print("Shape:", image.shape)
print("Height:", image.shape[0])
print("Width:", image.shape[1])
print("Channels:", image.shape[2])
print("Data type:", image.dtype)
print("Minimum pixel value:", image.min())
print("Maximum pixel value:", image.max())
print("Mean pixel value:", image.mean())