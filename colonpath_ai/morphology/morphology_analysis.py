from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import json
from skimage.measure import regionprops


# =========================================================
# PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "outputs" / "unet"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "morphology"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# SETTINGS
# =========================================================

MIN_GLAND_AREA = 50


# =========================================================
# CIRCULARITY
# =========================================================

def calculate_circularity(area, perimeter):
    """
    Circularity = 4*pi*Area / Perimeter^2

    Circle ≈ 1
    Irregular shapes < 1
    """

    if perimeter <= 0:
        return 0.0

    return (4.0 * np.pi * area) / (perimeter ** 2)


# =========================================================
# GLAND ANALYSIS
# =========================================================

def load_prediction_mask(mask_path):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise FileNotFoundError(
            f"Could not read prediction mask:\n{mask_path}"
        )

    _, binary = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary


def analyze_glands(mask):

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    measurements = []

    gland_number = 0

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < MIN_GLAND_AREA:
            continue

        gland_number += 1

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]

        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]

        gland_mask = np.zeros_like(mask)

        gland_mask[labels == label] = 255

        contours, _ = cv2.findContours(
            gland_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        perimeter = cv2.arcLength(
            contour,
            True
        )

        circularity = calculate_circularity(
            area,
            perimeter
        )

        aspect_ratio = (
            width / height
            if height != 0
            else 0.0
        )

        cx, cy = centroids[label]

        measurements.append({

            "gland_id": gland_number,

            "area_pixels": float(area),

            "perimeter_pixels": float(perimeter),

            "width_pixels": int(width),

            "height_pixels": int(height),

            "aspect_ratio": float(aspect_ratio),

            "circularity": float(circularity),

            "centroid_x": float(cx),

            "centroid_y": float(cy)
        })

    return measurements


# =========================================================
# GLAND VISUALIZATION
# =========================================================

def create_gland_visualization(
    mask_path,
    measurements
):

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        return None

    visualization = cv2.cvtColor(
        mask,
        cv2.COLOR_GRAY2BGR
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    gland_id = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_GLAND_AREA:
            continue

        gland_id += 1

        cv2.drawContours(
            visualization,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        M = cv2.moments(contour)

        if M["m00"] != 0:

            cx = int(
                M["m10"] / M["m00"]
            )

            cy = int(
                M["m01"] / M["m00"]
            )

            cv2.putText(
                visualization,
                str(gland_id),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

    return visualization


# =========================================================
# CONVERT HOVERNET CONTOUR
# =========================================================

def contour_to_numpy(contour):

    if contour is None:
        return None

    arr = np.asarray(
        contour,
        dtype=np.int32
    )

    # Handle [N, 1, 2]
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr.reshape(-1, 2)

    # Handle [N, 2]
    elif arr.ndim == 2 and arr.shape[1] == 2:
        pass

    else:
        return None

    if len(arr) < 3:
        return None

    return arr


# =========================================================
# NUCLEUS ANALYSIS
# =========================================================

def analyze_nuclei(json_path):

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    nuclei = data.get("nuc", {})

    measurements = []

    for nucleus_id, nucleus in nuclei.items():

        contour = nucleus.get(
            "contour",
            []
        )

        contour = contour_to_numpy(
            contour
        )

        if contour is None:
            continue

        # -------------------------------------------------
        # Area
        # -------------------------------------------------

        area = cv2.contourArea(
            contour
        )

        if area <= 0:
            continue

        # -------------------------------------------------
        # Perimeter
        # -------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True
        )

        # -------------------------------------------------
        # Circularity
        # -------------------------------------------------

        circularity = calculate_circularity(
            area,
            perimeter
        )

        # -------------------------------------------------
        # Eccentricity
        # -------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            contour
        )

        local_mask = np.zeros(
            (h + 2, w + 2),
            dtype=np.uint8
        )

        shifted = contour.copy()

        shifted[:, 0] -= x
        shifted[:, 1] -= y

        cv2.fillPoly(
            local_mask,
            [shifted],
            1
        )

        props = regionprops(
            local_mask
        )

        if props:

            eccentricity = float(
                props[0].eccentricity
            )

        else:

            eccentricity = 0.0

        # -------------------------------------------------
        # Centroid
        # -------------------------------------------------

        M = cv2.moments(contour)

        if M["m00"] != 0:

            centroid_x = (
                M["m10"] / M["m00"]
            )

            centroid_y = (
                M["m01"] / M["m00"]
            )

        else:

            centroid_x = 0.0
            centroid_y = 0.0

        # -------------------------------------------------
        # Type
        # -------------------------------------------------

        nucleus_type = nucleus.get(
            "type",
            None
        )

        measurements.append({

            "nucleus_id": str(nucleus_id),

            "type": nucleus_type,

            "area_px2": float(area),

            "perimeter_px": float(perimeter),

            "eccentricity": eccentricity,

            "circularity": float(circularity),

            "centroid_x": float(centroid_x),

            "centroid_y": float(centroid_y)
        })

    return measurements


# =========================================================
# SAVE NUCLEUS CSV
# =========================================================

def save_nucleus_csv(
    measurements,
    output_path
):

    if not measurements:

        print(
            "No valid nuclei found."
        )

        return

    dataframe = pd.DataFrame(
        measurements
    )

    dataframe.to_csv(
        output_path,
        index=False
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("COLON GLAND + NUCLEAR MORPHOLOGICAL ANALYSIS")
    print("=" * 60)

    print(
        f"Input directory : {INPUT_DIR}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()

    # =====================================================
    # GLAND ANALYSIS
    # =====================================================

    prediction_files = sorted(
        INPUT_DIR.glob("*_prediction.png")
    )

    print(
        f"Prediction images found: "
        f"{len(prediction_files)}"
    )

    all_gland_results = []

    for mask_path in prediction_files:

        print("-" * 60)

        print(
            f"Processing gland image: "
            f"{mask_path.name}"
        )

        mask = load_prediction_mask(
            mask_path
        )

        measurements = analyze_glands(
            mask
        )

        print(
            f"Glands detected: "
            f"{len(measurements)}"
        )

        for measurement in measurements:

            measurement["image"] = (
                mask_path.name
            )

            all_gland_results.append(
                measurement
            )

        visualization = (
            create_gland_visualization(
                mask_path,
                measurements
            )
        )

        if visualization is not None:

            visualization_path = (
                OUTPUT_DIR /
                f"{mask_path.stem}_morphology.png"
            )

            cv2.imwrite(
                str(visualization_path),
                visualization
            )

            print(
                f"Visualization saved: "
                f"{visualization_path}"
            )

    # =====================================================
    # SAVE GLAND CSV
    # =====================================================

    if all_gland_results:

        gland_df = pd.DataFrame(
            all_gland_results
        )

        gland_csv = (
            OUTPUT_DIR /
            "gland_measurements.csv"
        )

        gland_df.to_csv(
            gland_csv,
            index=False
        )

        print()
        print("GLAND SUMMARY")
        print("-" * 40)

        print(
            "Total glands:",
            len(gland_df)
        )

        print(
            "Mean area:",
            round(
                gland_df[
                    "area_pixels"
                ].mean(),
                2
            )
        )

        print(
            "Mean perimeter:",
            round(
                gland_df[
                    "perimeter_pixels"
                ].mean(),
                2
            )
        )

        print(
            "Mean circularity:",
            round(
                gland_df[
                    "circularity"
                ].mean(),
                4
            )
        )

        print(
            "Mean aspect ratio:",
            round(
                gland_df[
                    "aspect_ratio"
                ].mean(),
                4
            )
        )

        print(
            f"Saved: {gland_csv}"
        )

    # =====================================================
    # NUCLEAR ANALYSIS
    # =====================================================

    json_candidates = sorted(
        (
            PROJECT_ROOT /
            "outputs" /
            "hovernet_test" /
            "result" /
            "json"
        ).glob("*.json")
    )

    if not json_candidates:

        print()
        print(
            "No HoVer-Net JSON files found."
        )

    else:

        print()
        print(
            f"HoVer-Net JSON files found: "
            f"{len(json_candidates)}"
        )

        all_nucleus_results = []

        for json_path in json_candidates:

            print("-" * 60)

            print(
                f"Processing nuclei: "
                f"{json_path.name}"
            )

            nucleus_results = (
                analyze_nuclei(
                    json_path
                )
            )

            print(
                f"Nuclei measured: "
                f"{len(nucleus_results)}"
            )

            for result in nucleus_results:

                result["image"] = (
                    json_path.stem
                )

                all_nucleus_results.append(
                    result
                )

        if all_nucleus_results:

            nucleus_df = pd.DataFrame(
                all_nucleus_results
            )

            nucleus_csv = (
                OUTPUT_DIR /
                "nuclei_measurements.csv"
            )

            nucleus_df.to_csv(
                nucleus_csv,
                index=False
            )

            print()
            print("NUCLEAR SUMMARY")
            print("-" * 40)

            print(
                "Total nuclei:",
                len(nucleus_df)
            )

            print(
                "Mean area:",
                round(
                    nucleus_df[
                        "area_px2"
                    ].mean(),
                    2
                )
            )

            print(
                "Mean perimeter:",
                round(
                    nucleus_df[
                        "perimeter_px"
                    ].mean(),
                    2
                )
            )

            print(
                "Mean eccentricity:",
                round(
                    nucleus_df[
                        "eccentricity"
                    ].mean(),
                    3
                )
            )

            print(
                "Mean circularity:",
                round(
                    nucleus_df[
                        "circularity"
                    ].mean(),
                    3
                )
            )

            print(
                "Nuclei by type:"
            )

            print(
                nucleus_df[
                    "type"
                ].value_counts().to_dict()
            )

            print(
                f"Saved: {nucleus_csv}"
            )

    # =====================================================
    # COMPLETE
    # =====================================================

    print()
    print("=" * 60)
    print("MORPHOLOGICAL ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()