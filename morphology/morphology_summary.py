import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    required = [
        "nucleus_id",
        "type",
        "area_px2",
        "perimeter_px",
        "eccentricity",
        "circularity",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Make sure numeric columns are actually numeric
    numeric_cols = [
        "area_px2",
        "perimeter_px",
        "eccentricity",
        "circularity",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    type_counts = {
        str(k): int(v)
        for k, v in df["type"].value_counts().to_dict().items()
    }

    summary = {
        "nuclei_measured": int(len(df)),

        "area_px2": {
            "mean": float(df["area_px2"].mean()),
            "median": float(df["area_px2"].median()),
            "std": float(df["area_px2"].std()),
            "min": float(df["area_px2"].min()),
            "max": float(df["area_px2"].max()),
        },

        "perimeter_px": {
            "mean": float(df["perimeter_px"].mean()),
            "median": float(df["perimeter_px"].median()),
            "std": float(df["perimeter_px"].std()),
            "min": float(df["perimeter_px"].min()),
            "max": float(df["perimeter_px"].max()),
        },

        "eccentricity": {
            "mean": float(df["eccentricity"].mean()),
            "median": float(df["eccentricity"].median()),
            "std": float(df["eccentricity"].std()),
        },

        "circularity": {
            "mean": float(df["circularity"].mean()),
            "median": float(df["circularity"].median()),
            "std": float(df["circularity"].std()),
        },

        "nuclei_by_type": type_counts,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== MORPHOLOGY SUMMARY ===")
    print("Nuclei measured:", summary["nuclei_measured"])
    print("Mean area:", round(summary["area_px2"]["mean"], 2))
    print("Mean perimeter:", round(summary["perimeter_px"]["mean"], 2))
    print("Mean eccentricity:", round(summary["eccentricity"]["mean"], 3))
    print("Mean circularity:", round(summary["circularity"]["mean"], 3))
    print("Nuclei by type:", summary["nuclei_by_type"])
    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()