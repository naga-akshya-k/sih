import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Combine nuclear and gland morphology results"
    )

    parser.add_argument("--nuclei", required=True)
    parser.add_argument("--glands", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    nuclei_df = pd.read_csv(args.nuclei)
    glands_df = pd.read_csv(args.glands)

    # ============================================================
    # NUCLEAR ANALYSIS
    # ============================================================

    required_nuclei = [
        "nucleus_id",
        "type",
        "area_px2",
        "perimeter_px",
        "eccentricity",
        "circularity",
    ]

    missing_nuclei = [
        c for c in required_nuclei
        if c not in nuclei_df.columns
    ]

    if missing_nuclei:
        raise ValueError(
            f"Missing nuclear columns: {missing_nuclei}"
        )

    nuclei = {
        "total": int(len(nuclei_df)),

        "types": {
            str(k): int(v)
            for k, v in nuclei_df["type"]
            .value_counts()
            .to_dict()
            .items()
        },

        "mean_area_px2": float(
            nuclei_df["area_px2"].mean()
        ),

        "mean_perimeter_px": float(
            nuclei_df["perimeter_px"].mean()
        ),

        "mean_eccentricity": float(
            nuclei_df["eccentricity"].mean()
        ),

        "mean_circularity": float(
            nuclei_df["circularity"].mean()
        ),
    }

    # ============================================================
    # GLAND ANALYSIS
    # ============================================================

    required_glands = [
        "gland_id",
        "area_pixels",
        "perimeter_pixels",
        "width_pixels",
        "height_pixels",
        "aspect_ratio",
        "circularity",
    ]

    missing_glands = [
        c for c in required_glands
        if c not in glands_df.columns
    ]

    if missing_glands:
        raise ValueError(
            f"Missing gland columns: {missing_glands}"
        )

    glands = {
        "total": int(len(glands_df)),

        "mean_area_pixels": float(
            glands_df["area_pixels"].mean()
        ),

        "mean_perimeter_pixels": float(
            glands_df["perimeter_pixels"].mean()
        ),

        "mean_width_pixels": float(
            glands_df["width_pixels"].mean()
        ),

        "mean_height_pixels": float(
            glands_df["height_pixels"].mean()
        ),

        "mean_aspect_ratio": float(
            glands_df["aspect_ratio"].mean()
        ),

        "mean_circularity": float(
            glands_df["circularity"].mean()
        ),
    }

    # ============================================================
    # COMBINED CASE SUMMARY
    # ============================================================

    case_summary = {
        "case_id": Path(args.nuclei).stem.replace(
            "_nuclei", ""
        ),

        "nuclei": nuclei,

        "glands": glands,
    }

    # ============================================================
    # SAVE JSON
    # ============================================================

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            case_summary,
            f,
            indent=2
        )

    # ============================================================
    # TERMINAL SUMMARY
    # ============================================================

    print("=" * 60)
    print("COLON HISTOPATHOLOGY CASE SUMMARY")
    print("=" * 60)

    print("\nNUCLEAR ANALYSIS")
    print("-" * 30)

    print(
        "Total nuclei:",
        nuclei["total"]
    )

    print(
        "Nuclei by type:",
        nuclei["types"]
    )

    print(
        "Mean area:",
        round(nuclei["mean_area_px2"], 2),
        "px²"
    )

    print(
        "Mean perimeter:",
        round(nuclei["mean_perimeter_px"], 2),
        "px"
    )

    print(
        "Mean eccentricity:",
        round(nuclei["mean_eccentricity"], 3)
    )

    print(
        "Mean circularity:",
        round(nuclei["mean_circularity"], 3)
    )

    print("\nGLAND ANALYSIS")
    print("-" * 30)

    print(
        "Total glands:",
        glands["total"]
    )

    print(
        "Mean area:",
        round(glands["mean_area_pixels"], 2),
        "pixels"
    )

    print(
        "Mean perimeter:",
        round(glands["mean_perimeter_pixels"], 2),
        "pixels"
    )

    print(
        "Mean width:",
        round(glands["mean_width_pixels"], 2),
        "pixels"
    )

    print(
        "Mean height:",
        round(glands["mean_height_pixels"], 2),
        "pixels"
    )

    print(
        "Mean aspect ratio:",
        round(glands["mean_aspect_ratio"], 3)
    )

    print(
        "Mean circularity:",
        round(glands["mean_circularity"], 3)
    )

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    main()