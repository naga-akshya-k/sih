import json
import csv
import argparse
from pathlib import Path

import numpy as np
import cv2


def analyze(json_path, output_dir):
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    nuclei = data["nuc"]

    rows = []

    for nucleus_id, nucleus in nuclei.items():

        contour = nucleus.get("contour", [])
        nucleus_type = nucleus.get("type", None)

        if len(contour) < 3:
            continue

        contour = np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Eccentricity from fitted ellipse
        eccentricity = np.nan

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)

            (_, _), (major_axis, minor_axis), _ = ellipse

            major_axis = max(major_axis, minor_axis)
            minor_axis = min(major_axis, minor_axis)

            if major_axis > 0:
                ratio = (minor_axis / major_axis) ** 2
                eccentricity = np.sqrt(max(0, 1 - ratio))

        # Circularity
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = np.nan

        rows.append({
            "nucleus_id": nucleus_id,
            "type": nucleus_type,
            "area_px2": area,
            "perimeter_px": perimeter,
            "eccentricity": eccentricity,
            "circularity": circularity
        })

    # -------------------------
    # Save per-nucleus CSV
    # -------------------------

    csv_path = output_dir / f"{json_path.stem}_nuclei.csv"

    fieldnames = [
        "nucleus_id",
        "type",
        "area_px2",
        "perimeter_px",
        "eccentricity",
        "circularity"
    ]

    with open(csv_path, "w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # -------------------------
    # Summary statistics
    # -------------------------

    areas = np.array(
        [r["area_px2"] for r in rows],
        dtype=float
    )

    perimeters = np.array(
        [r["perimeter_px"] for r in rows],
        dtype=float
    )

    eccentricities = np.array(
        [
            r["eccentricity"]
            for r in rows
            if not np.isnan(r["eccentricity"])
        ],
        dtype=float
    )

    circularities = np.array(
        [
            r["circularity"]
            for r in rows
            if not np.isnan(r["circularity"])
        ],
        dtype=float
    )

    summary = {
        "image": json_path.stem,
        "nuclei_measured": len(rows),

        "mean_area_px2": float(np.mean(areas)),
        "median_area_px2": float(np.median(areas)),

        "mean_perimeter_px": float(np.mean(perimeters)),
        "median_perimeter_px": float(np.median(perimeters)),

        "mean_eccentricity": float(np.mean(eccentricities)),
        "mean_circularity": float(np.mean(circularities))
    }

    summary_path = output_dir / f"{json_path.stem}_summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("========== MORPHOLOGY ==========")
    print(f"Nuclei measured:     {len(rows)}")
    print(f"Mean area:           {summary['mean_area_px2']:.2f} px²")
    print(f"Mean perimeter:      {summary['mean_perimeter_px']:.2f} px")
    print(f"Mean eccentricity:   {summary['mean_eccentricity']:.3f}")
    print(f"Mean circularity:    {summary['mean_circularity']:.3f}")
    print()
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    analyze(
        args.json,
        args.output
    )