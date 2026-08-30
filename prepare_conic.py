from pathlib import Path
import glob
import io

import pandas as pd
from PIL import Image
import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

INPUT_DIR = PROJECT_DIR / "datasets" / "conic2022" / "data"

OUTPUT_DIR = PROJECT_DIR / "datasets" / "conic2022_processed"

IMAGE_DIR = OUTPUT_DIR / "images"
INST_DIR = OUTPUT_DIR / "inst_maps"
CLASS_DIR = OUTPUT_DIR / "class_maps"


# ---------------------------------------------------------
# CREATE OUTPUT DIRECTORIES
# ---------------------------------------------------------

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
INST_DIR.mkdir(parents=True, exist_ok=True)
CLASS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# FIND ALL PARQUET FILES
# ---------------------------------------------------------

files = sorted(INPUT_DIR.glob("*.parquet"))

print("=" * 60)
print("CoNIC2022 PREPARATION")
print("=" * 60)

print(f"Input directory : {INPUT_DIR}")
print(f"Parquet files   : {len(files)}")

if not files:
    raise RuntimeError("No parquet files found!")


# ---------------------------------------------------------
# HELPER: EXTRACT BYTES
# ---------------------------------------------------------

def get_bytes(value):
    """
    CoNIC stores image/mask data as dictionaries containing
    a 'bytes' field.
    """
    if isinstance(value, dict):
        return value["bytes"]

    if isinstance(value, bytes):
        return value

    raise TypeError(f"Unexpected data type: {type(value)}")


# ---------------------------------------------------------
# PROCESS EACH PARQUET
# ---------------------------------------------------------

global_index = 0

for parquet_file in files:

    print()
    print("-" * 60)
    print(f"Reading: {parquet_file.name}")

    df = pd.read_parquet(parquet_file)

    print(f"Rows: {len(df)}")

    for _, row in df.iterrows():

        # -------------------------------------------------
        # IMAGE
        # -------------------------------------------------

        image_bytes = get_bytes(row["image"])

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_path = IMAGE_DIR / f"{global_index:05d}.png"
        image.save(image_path)


        # -------------------------------------------------
        # INSTANCE MAP
        # -------------------------------------------------

        inst_bytes = get_bytes(row["inst_map"])

        inst_image = Image.open(io.BytesIO(inst_bytes))

        inst_array = np.array(inst_image)

        inst_path = INST_DIR / f"{global_index:05d}.npy"
        np.save(inst_path, inst_array)


        # -------------------------------------------------
        # CLASS MAP
        # -------------------------------------------------

        class_bytes = get_bytes(row["class_map"])

        class_image = Image.open(io.BytesIO(class_bytes))

        class_array = np.array(class_image)

        class_path = CLASS_DIR / f"{global_index:05d}.npy"
        np.save(class_path, class_array)


        # -------------------------------------------------
        # PROGRESS
        # -------------------------------------------------

        global_index += 1

        if global_index % 100 == 0:
            print(f"Processed: {global_index}")


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Total samples: {global_index}")

print()
print("Images     :", len(list(IMAGE_DIR.glob("*.png"))))
print("Inst maps  :", len(list(INST_DIR.glob("*.npy"))))
print("Class maps :", len(list(CLASS_DIR.glob("*.npy"))))

print()
print("Output:")
print(OUTPUT_DIR)