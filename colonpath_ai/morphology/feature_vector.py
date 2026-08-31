import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        case = json.load(f)

    nuclei = case["nuclei"]
    glands = case["glands"]

    features = {
        "case_id": case["case_id"],

        # Nuclear features
        "nuclei_total": nuclei["total"],
        "nuclei_type_1": nuclei["types"].get("1", 0),
        "nuclei_type_2": nuclei["types"].get("2", 0),
        "nuclei_type_3": nuclei["types"].get("3", 0),
        "nuclei_type_4": nuclei["types"].get("4", 0),
        "nuclei_mean_area_px2": nuclei["mean_area_px2"],
        "nuclei_mean_perimeter_px": nuclei["mean_perimeter_px"],
        "nuclei_mean_eccentricity": nuclei["mean_eccentricity"],
        "nuclei_mean_circularity": nuclei["mean_circularity"],

        # Gland features
        "glands_total": glands["total"],
        "glands_mean_area_px2": glands["mean_area_pixels"],
        "glands_mean_perimeter_px": glands["mean_perimeter_pixels"],
        "glands_mean_width_px": glands["mean_width_pixels"],
        "glands_mean_height_px": glands["mean_height_pixels"],
        "glands_mean_aspect_ratio": glands["mean_aspect_ratio"],
        "glands_mean_circularity": glands["mean_circularity"],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    print("=" * 60)
    print("AI FEATURE VECTOR CREATED")
    print("=" * 60)

    for key, value in features.items():
        print(f"{key}: {value}")

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()